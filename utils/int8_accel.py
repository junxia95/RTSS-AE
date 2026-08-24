import copy

import torch
import torch.nn as nn


class _OutputOnlyWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, dict):
            return output['output']
        return output


def _safe_backend_name(name):
    return str(name or 'none').strip().lower()


def _try_tensorrt(model, sample_input):
    try:
        import torch_tensorrt  # noqa: F401
    except Exception:
        return None

    try:
        import torch_tensorrt
        wrapper = _OutputOnlyWrapper(copy.deepcopy(model).eval().cpu())
        if not isinstance(sample_input, torch.Tensor):
            return None
        sample_input = sample_input.detach().cpu().contiguous()
        compiled = torch_tensorrt.compile(
            wrapper,
            inputs=[
                torch_tensorrt.Input(
                    min_shape=(1, *sample_input.shape[1:]),
                    opt_shape=tuple(sample_input.shape),
                    max_shape=(max(sample_input.shape[0], 1), *sample_input.shape[1:]),
                    dtype=torch.float32,
                )
            ],
            enabled_precisions={torch.int8},
        )
        return compiled.to('cuda' if torch.cuda.is_available() else 'cpu')
    except Exception:
        return None


def _try_fx_int8(model, sample_input, calibration_loader, calib_batches):
    try:
        from torch.ao.quantization import get_default_qconfig_mapping
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    except Exception:
        return None

    try:
        wrapper = _OutputOnlyWrapper(copy.deepcopy(model).eval().cpu())
        sample_input = sample_input.detach().cpu().contiguous()
        qconfig_mapping = get_default_qconfig_mapping('fbgemm')
        prepared = prepare_fx(wrapper, qconfig_mapping, (sample_input,))
        prepared.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                if batch_idx >= calib_batches:
                    break
                prepared(data.detach().cpu())
        quantized = convert_fx(prepared)
        return quantized.cpu()
    except Exception:
        return None


def build_int8_eval_model(model, sample_input, calibration_loader=None, backend='none', calib_batches=8):
    backend = _safe_backend_name(backend)
    if backend == 'none':
        return model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 'none'

    if backend == 'tensorrt':
        compiled = _try_tensorrt(model, sample_input)
        if compiled is not None:
            return compiled, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 'tensorrt'
        backend = 'fx_int8'

    if backend in ('fx_int8', 'int8', 'torch_int8', 'auto'):
        if calibration_loader is None:
            return model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 'none'
        quantized = _try_fx_int8(model, sample_input, calibration_loader, calib_batches)
        if quantized is not None:
            return quantized, torch.device('cpu'), 'fx_int8'

    return model, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 'none'


def extract_logits(output):
    if isinstance(output, dict):
        return output['output']
    return output
