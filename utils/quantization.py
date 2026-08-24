import lzma
import zlib

import numpy as np
import torch
import torch.nn.functional as F

MIN_QUANT_BITS = 8

_TORCH_TO_NUMPY_DTYPE = {
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.float64: np.float64,
    torch.int64: np.int64,
    torch.int32: np.int32,
    torch.int16: np.int16,
    torch.int8: np.int8,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}

_DTYPE_NAME_TO_TORCH = {
    'float32': torch.float32,
    'float16': torch.float16,
    'float64': torch.float64,
    'int64': torch.int64,
    'int32': torch.int32,
    'int16': torch.int16,
    'int8': torch.int8,
    'uint8': torch.uint8,
    'bool': torch.bool,
}


def _dtype_name(dtype):
    return str(dtype).replace('torch.', '')


def _metadata_nbytes(name, shape, payload_dtype, codec='none'):
    return (
        len(str(name).encode('utf-8'))
        + len(str(payload_dtype).encode('utf-8'))
        + len(str(codec).encode('utf-8'))
        + 8 * len(shape)
        + 40
    )


def effective_quant_bits(bits):
    if bits is None:
        return bits
    return max(int(bits), MIN_QUANT_BITS)


def _resolve_bits_for_name(name, bits):
    """bits: int or callable(name_str) -> int."""
    if callable(bits):
        return int(bits(str(name)))
    return int(bits)


def state_dict_weighted_mean_bits(model_or_state_dict, bits, enabled=True):
    """Numel-weighted mean effective bit width (used as a scalar comm proxy)."""
    if not enabled:
        return float(max(int(MIN_QUANT_BITS), 32))
    if hasattr(model_or_state_dict, 'state_dict'):
        items = model_or_state_dict.state_dict().items()
    else:
        items = model_or_state_dict.items()
    total = 0
    w_bits = 0.0
    for name, tensor in items:
        n = int(tensor.numel())
        if n <= 0:
            continue
        b = _resolve_bits_for_name(name, bits)
        eff = float(effective_quant_bits(b)) if b < 32 else 32.0
        w_bits += eff * n
        total += n
    if total <= 0:
        return float(effective_quant_bits(_resolve_bits_for_name('', bits)))
    return w_bits / float(total)


def fake_quant_tensor(tensor, bits):
    if bits is None or bits <= 0:
        return tensor
    bits = effective_quant_bits(bits)
    if bits >= 32:
        return tensor

    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    if qmax <= 0:
        return tensor

    max_abs = tensor.detach().abs().max()
    if max_abs.item() == 0:
        return tensor

    scale = max_abs / qmax
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax) * scale
    return tensor + (quantized - tensor).detach()


def _normalize_comm_8bit_format(fmt):
    """通信侧 8-bit 载荷：int8(对称+scale) 或 PyTorch FP8。"""
    s = 'int8' if fmt is None else str(fmt).strip().lower()
    if s in ('fp8', 'fp8_e4m3', 'fp8_e4m3fn', 'e4m3', 'e4m3fn'):
        return 'fp8_e4m3fn'
    if s in ('fp8_e5m2', 'e5m2'):
        return 'fp8_e5m2'
    if s in ('int8', 'i8', 'symint8'):
        return 'int8'
    return 'int8'


def _fp8_dtype_for_mode(mode):
    if mode == 'fp8_e4m3fn':
        return getattr(torch, 'float8_e4m3fn', None)
    if mode == 'fp8_e5m2':
        return getattr(torch, 'float8_e5m2', None)
    return None


def _fp8_payload_dtype_string(mode):
    if mode == 'fp8_e4m3fn':
        return 'float8_e4m3fn'
    if mode == 'fp8_e5m2':
        return 'float8_e5m2'
    return None


def fake_quant_tensor_comm_matched(tensor, bits, comm_8bit_format='int8'):
    """
    与通信 pack 对齐的 STE 伪量化：8-bit 可选 FP8 圆整，9~16-bit 用 float16 圆整，否则对称整型伪量化。
    """
    if bits is None or bits <= 0:
        return tensor
    bits = effective_quant_bits(bits)
    if bits >= 32:
        return tensor
    mode8 = _normalize_comm_8bit_format(comm_8bit_format)
    if bits <= 8:
        dt = _fp8_dtype_for_mode(mode8)
        if mode8 != 'int8' and dt is not None:
            try:
                xf = tensor.float()
                q = xf.to(dt).float()
                return tensor + (q - tensor).detach()
            except (RuntimeError, TypeError):
                pass
        return fake_quant_tensor(tensor, bits)
    if bits <= 16:
        xf = tensor.float()
        q = xf.to(torch.float16).float()
        return tensor + (q - tensor).detach()
    return fake_quant_tensor(tensor, bits)


def fake_quant_tensor_ste(tensor, bits):
    """张量级 STE 伪量化（用于激活或通用张量），尺度基于张量整体 max-abs。"""
    if bits is None or bits <= 0 or bits >= 32:
        return tensor
    bits = effective_quant_bits(bits)
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    if qmax <= 0:
        return tensor
    max_abs = tensor.abs().max().detach().clamp(min=1e-12)
    scale = max_abs / float(qmax)
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax) * scale
    return tensor + (quantized - tensor).detach()


def fake_quant_weight_channelwise_ste(weight, bits):
    """Conv:[O,I,k,k]、Linear:[O,I] 按输出通道（输出滤波器）独立尺度 + STE；其它形状退回张量级。"""
    if bits is None or bits <= 0 or bits >= 32:
        return weight
    bits = effective_quant_bits(bits)
    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    if qmax <= 0:
        return weight
    if weight.ndim == 4:
        reduce_dims = (1, 2, 3)
    elif weight.ndim == 2:
        reduce_dims = (1,)
    else:
        return fake_quant_tensor_ste(weight, bits)
    max_abs = weight.abs().amax(dim=reduce_dims, keepdim=True).detach().clamp(min=1e-12)
    scale = max_abs / float(qmax)
    quantized = torch.clamp(torch.round(weight / scale), qmin, qmax) * scale
    return weight + (quantized - weight).detach()


def maybe_quantize_batch(tensor, bits, enabled, channelwise_act=False):
    if not enabled:
        return tensor
    if channelwise_act and tensor.dim() == 4:
        bits_eff = effective_quant_bits(bits)
        if bits_eff >= 32:
            return tensor
        qmax = (2 ** (bits_eff - 1)) - 1
        qmin = -(2 ** (bits_eff - 1))
        max_abs = tensor.abs().amax(dim=(0, 2, 3), keepdim=True).detach().clamp(min=1e-12)
        scale = max_abs / float(qmax)
        q = torch.clamp(torch.round(tensor / scale), qmin, qmax) * scale
        return tensor + (q - tensor).detach()
    return fake_quant_tensor(tensor, bits)


def quantized_state_dict(model, bits, enabled, comm_8bit_format='int8'):
    return unpack_state_dict_payload(
        pack_state_dict_payload(model, bits, enabled, comm_8bit_format=comm_8bit_format)
    )


def _normalize_codec(codec):
    codec = 'none' if codec is None else str(codec).strip().lower()
    if codec in ('none', 'zlib', 'lzma'):
        return codec
    return 'none'


def _compress_lossless(data, codec='none', compression_level=6):
    codec = _normalize_codec(codec)
    if codec == 'none' or len(data) == 0:
        return data, 'none', False

    level = int(compression_level) if compression_level is not None else 6
    if codec == 'zlib':
        level = max(0, min(level, 9))
        compressed = zlib.compress(data, level)
    elif codec == 'lzma':
        level = max(0, min(level, 9))
        compressed = lzma.compress(data, preset=level)
    else:
        return data, 'none', False

    if len(compressed) >= len(data):
        return data, 'none', False
    return compressed, codec, True


def _decompress_lossless(data, codec='none', encoded=False):
    codec = _normalize_codec(codec)
    if not encoded or codec == 'none':
        return data
    if codec == 'zlib':
        return zlib.decompress(data)
    if codec == 'lzma':
        return lzma.decompress(data)
    return data


@torch.no_grad()
def project_model_to_quantization(model, bits, enabled, comm_8bit_format='int8'):
    if not enabled:
        return
    for name, param in model.named_parameters():
        b = _resolve_bits_for_name(name, bits)
        param.copy_(fake_quant_tensor_comm_matched(param, b, comm_8bit_format))


def apply_conv_linear_qat_forward(net, bits_policy, enabled=True):
    """
    在 Conv2d / Linear 前向中插入通道级权重伪量化（STE），与分层 bits_policy 兼容。
    返回 (module, original_forward) 列表，供训练结束后 restore。
    """
    if not enabled:
        return []
    from torch.nn import Conv2d, Linear

    restored = []
    for name, module in net.named_modules():
        if isinstance(module, Conv2d):
            orig_forward = module.forward
            stride = module.stride
            padding = module.padding
            dilation = module.dilation
            groups = module.groups

            def _make_conv(name_, mod, stride_, padding_, dilation_, groups_):
                def _forward(x):
                    b = _resolve_bits_for_name(name_, bits_policy)
                    wq = fake_quant_weight_channelwise_ste(mod.weight, b)
                    return F.conv2d(x, wq, mod.bias, stride_, padding_, dilation_, groups_)

                return _forward

            module.forward = _make_conv(name, module, stride, padding, dilation, groups)
            restored.append((module, orig_forward))
        elif isinstance(module, Linear):
            orig_forward = module.forward

            def _make_lin(name_, mod):
                def _forward(x):
                    b = _resolve_bits_for_name(name_, bits_policy)
                    wq = fake_quant_weight_channelwise_ste(mod.weight, b)
                    return F.linear(x, wq, mod.bias)

                return _forward

            module.forward = _make_lin(name, module)
            restored.append((module, orig_forward))
    return restored


def restore_conv_linear_qat_forward(restored):
    for module, orig_forward in restored:
        module.forward = orig_forward


def get_quant_comm_ratio(bits, enabled, base_bits=32):
    if not enabled:
        return 1.0
    if bits is None:
        return 1.0
    if isinstance(bits, float):
        eff = max(bits, 1.0)
    else:
        eff = float(effective_quant_bits(int(bits)))
    return eff / max(float(base_bits), 1.0)


def pack_tensor_payload(
    name,
    tensor,
    bits,
    enabled=True,
    codec='none',
    compression_level=6,
    comm_8bit_format='int8',
):
    value = tensor.detach().cpu().contiguous()
    shape = tuple(value.shape)
    original_dtype = _dtype_name(value.dtype)
    bits = effective_quant_bits(bits)
    codec = _normalize_codec(codec)
    mode8 = _normalize_comm_8bit_format(comm_8bit_format)

    if enabled and torch.is_floating_point(value) and bits <= 8:
        float_value = value.float()
        fp8_dt = _fp8_dtype_for_mode(mode8)
        use_int8_pack = mode8 == 'int8' or fp8_dt is None
        if not use_int8_pack:
            try:
                packed_fp8 = float_value.to(fp8_dt).contiguous()
                u8 = packed_fp8.view(torch.uint8)
                payload_dtype = _fp8_payload_dtype_string(mode8)
                data = u8.numpy().tobytes()
                scale_value = None
            except (RuntimeError, TypeError):
                use_int8_pack = True
        if use_int8_pack:
            max_abs = float_value.abs().max()
            scale = (max_abs / 127.0).item() if max_abs.item() > 0 else 1.0
            packed = torch.clamp(torch.round(float_value / scale), -128, 127).to(torch.int8).numpy()
            payload_dtype = 'int8'
            data = packed.tobytes()
            scale_value = float(scale)
    elif enabled and torch.is_floating_point(value) and bits <= 16:
        payload_dtype = 'float16'
        data = value.float().to(torch.float16).numpy().tobytes()
        scale_value = None
    else:
        payload_dtype = original_dtype
        np_dtype = _TORCH_TO_NUMPY_DTYPE.get(value.dtype)
        if np_dtype is None:
            value = value.float()
            payload_dtype = 'float32'
            np_dtype = np.float32
        data = value.numpy().astype(np_dtype, copy=False).tobytes()
        scale_value = None

    raw_data_bytes = len(data)
    codec_used = 'none'
    encoded = False
    if codec != 'none':
        data, codec_used, encoded = _compress_lossless(data, codec, compression_level=compression_level)

    metadata_bytes = _metadata_nbytes(name, shape, payload_dtype, codec_used)
    return {
        'name': str(name),
        'shape': shape,
        'original_dtype': original_dtype,
        'payload_dtype': payload_dtype,
        'bits': int(bits),
        'scale': scale_value,
        'data': data,
        'codec': codec_used,
        'encoded': bool(encoded),
        'compression_level': int(compression_level),
        'uncompressed_data_bytes': int(raw_data_bytes),
        'data_bytes': int(len(data)),
        'metadata_bytes': int(metadata_bytes),
        'payload_bytes': int(len(data) + metadata_bytes),
    }


def unpack_tensor_payload(entry):
    payload_dtype = entry['payload_dtype']
    shape = tuple(entry['shape'])
    original_dtype = _DTYPE_NAME_TO_TORCH.get(entry['original_dtype'], torch.float32)
    data = _decompress_lossless(entry['data'], entry.get('codec', 'none'), entry.get('encoded', False))

    if payload_dtype == 'int8':
        array = np.frombuffer(data, dtype=np.int8).copy()
        tensor = torch.from_numpy(array).view(shape).float() * float(entry.get('scale') or 1.0)
        return tensor.to(original_dtype if torch.is_floating_point(torch.empty((), dtype=original_dtype)) else torch.float32)

    if payload_dtype in ('float8_e4m3fn', 'float8_e5m2'):
        dt = getattr(torch, payload_dtype, None)
        if dt is None:
            np_dtype = np.float32
            array = np.frombuffer(data, dtype=np_dtype).copy()
            tensor = torch.from_numpy(array).view(shape)
            return tensor.to(original_dtype)
        u8 = torch.frombuffer(bytearray(data), dtype=torch.uint8).reshape(shape).contiguous()
        tensor = u8.view(dt).float()
        return tensor.to(original_dtype if torch.is_floating_point(torch.empty((), dtype=original_dtype)) else torch.float32)

    np_dtype = getattr(np, payload_dtype, None)
    if np_dtype is None:
        np_dtype = np.float32
    array = np.frombuffer(data, dtype=np_dtype).copy()
    tensor = torch.from_numpy(array).view(shape)
    return tensor.to(original_dtype)


def _iter_state_dict(model_or_state_dict):
    if hasattr(model_or_state_dict, 'state_dict'):
        return model_or_state_dict.state_dict().items()
    return model_or_state_dict.items()


def pack_state_dict_payload(
    model_or_state_dict,
    bits,
    enabled=True,
    codec='none',
    compression_level=6,
    comm_8bit_format='int8',
):
    entries = []
    for name, tensor in _iter_state_dict(model_or_state_dict):
        b = _resolve_bits_for_name(name, bits)
        entries.append(
            pack_tensor_payload(
                name,
                tensor,
                b,
                enabled=enabled,
                codec=codec,
                compression_level=compression_level,
                comm_8bit_format=comm_8bit_format,
            )
        )
    if callable(bits):
        mean_b = state_dict_weighted_mean_bits(model_or_state_dict, bits, enabled=enabled)
        bits_meta = int(max(MIN_QUANT_BITS, min(32, round(mean_b))))
    else:
        bits_meta = int(effective_quant_bits(bits))
    payload_bytes = sum(entry['payload_bytes'] for entry in entries)
    data_bytes = sum(entry['data_bytes'] for entry in entries)
    uncompressed_data_bytes = sum(entry.get('uncompressed_data_bytes', entry['data_bytes']) for entry in entries)
    metadata_bytes = sum(entry['metadata_bytes'] for entry in entries)
    return {
        'bits': bits_meta,
        'enabled': bool(enabled),
        'codec': _normalize_codec(codec),
        'compression_level': int(compression_level),
        'entries': entries,
        'payload_bytes': int(payload_bytes),
        'data_bytes': int(data_bytes),
        'uncompressed_data_bytes': int(uncompressed_data_bytes),
        'metadata_bytes': int(metadata_bytes),
        'tensor_count': int(len(entries)),
    }


def unpack_state_dict_payload(payload):
    state_dict = {}
    for entry in payload['entries']:
        state_dict[entry['name']] = unpack_tensor_payload(entry)
    return state_dict


def transmit_state_dict(
    model_or_state_dict,
    bits,
    enabled=True,
    codec='none',
    compression_level=6,
    comm_8bit_format='int8',
):
    payload = pack_state_dict_payload(
        model_or_state_dict,
        bits,
        enabled=enabled,
        codec=codec,
        compression_level=compression_level,
        comm_8bit_format=comm_8bit_format,
    )
    mean_bits = float(state_dict_weighted_mean_bits(model_or_state_dict, bits, enabled=enabled))
    return unpack_state_dict_payload(payload), payload['payload_bytes'], {
        'bits': int(payload['bits']),
        'mean_bits': mean_bits,
        'codec': payload['codec'],
        'compression_level': int(payload['compression_level']),
        'payload_bytes': int(payload['payload_bytes']),
        'data_bytes': int(payload['data_bytes']),
        'uncompressed_data_bytes': int(payload['uncompressed_data_bytes']),
        'metadata_bytes': int(payload['metadata_bytes']),
        'tensor_count': int(payload['tensor_count']),
    }


def state_dict_payload_nbytes(
    model_or_state_dict,
    bits,
    enabled=True,
    codec='none',
    compression_level=6,
    comm_8bit_format='int8',
):
    return int(
        pack_state_dict_payload(
            model_or_state_dict,
            bits,
            enabled=enabled,
            codec=codec,
            compression_level=compression_level,
            comm_8bit_format=comm_8bit_format,
        )['payload_bytes']
    )
