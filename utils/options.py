#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.2, help="the fraction of clients: C")
    parser.add_argument('--baseline_active_sampling', type=int, default=0,
                        help='when 1, legacy baselines train/communicate only int(num_users*frac) sampled clients per round')
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    parser.add_argument('--personal_ep', type=int, default=1, help="the number of personal model epochs: E")
    parser.add_argument('--shared_ep', type=int, default=5, help="the number of shared model epochs: E")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=50, help="test batch size")
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.998, help="learning rate decay")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="weight_decay (default: 1e-4)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument("--algorithm", type=str, default="DFL_MM")
    parser.add_argument("--cifar100_coarse", type=int, default=0, help="use 20 class cifar100")

    # model arguments
    parser.add_argument('--model', type=str, default='resnet8', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--use_project_head', type=int, default=0)
    parser.add_argument('--out_dim', type=int, default=256, help='the output dimension for the projection layer')

    # other arguments
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--data_root', type=str, default='./data',
                        help='root directory for raw datasets; partition json files stay under ./data')
    parser.add_argument('--generate_data', type=int, default=0, help="whether generate new dataset")
    parser.add_argument('--iid', type=int, default=0, help='whether i.i.d or not')
    parser.add_argument('--noniid_case', type=int, default=5, help="non i.i.d case (1, 2, 3, 4)")
    parser.add_argument('--data_beta', type=float, default=1.0,
                        help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--partition_tag', type=str, default='',
                        help='optional suffix for generated/read data split json files')
    parser.add_argument('--experiment_tag', type=str, default='',
                        help='stable experiment variant name recorded with result metadata')
    parser.add_argument('--result_dir', type=str, default='',
                        help='optional per-run result directory; avoids timestamp-based output collisions')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    parser.add_argument('--num_sizes', type=int, default=32, help="number of sizes of images")
    parser.add_argument('--max_train_samples', type=int, default=0,
                        help='limit training samples for quick smoke runs; 0 means full dataset')
    parser.add_argument('--max_test_samples', type=int, default=0,
                        help='limit test samples for quick smoke runs; 0 means full dataset')
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    # topo structure
    parser.add_argument('--topo', type=str, default='M-ring',
                        help="topo structure of communication: ring, M-ring, full, star, random, cluster, clustered")
    parser.add_argument('--client_selection', type=str, default='comprehensive',
                        help='strategy of client selection: data_aware, speed_aware, forget_aware, comprehensive, random, rl, rl_aggregation')
    parser.add_argument('--weight_data', type=float, default=0.8, help='weight of data aware')
    parser.add_argument('--weight_speed', type=float, default=0.0, help='weight of speed aware')
    parser.add_argument('--weight_forget', type=float, default=0.2, help='weight of forget aware')
    parser.add_argument('--curiosity', type=float, default=0.2, help='The probability of violating the client selection strategy')
    parser.add_argument('--aggregation', type=bool, default=True, help='whether aggregation when different model meeting')
    parser.add_argument('--device_freq_jitter', type=float, default=0.05,
                        help='relative jitter for per-client device frequency')
    parser.add_argument('--device_power_jitter', type=float, default=0.05,
                        help='relative jitter for per-client device power')
    parser.add_argument('--device_battery_jitter', type=float, default=0.05,
                        help='relative jitter for per-client battery capacity')
    parser.add_argument('--dfl_battery_capacity_scale', type=float, default=1.0,
                        help='DFL_MM only: multiply each client battery capacity before training')
    parser.add_argument('--dfl_battery_energy_scale', type=float, default=1.0,
                        help='DFL_MM only: multiply training/communication energy drain')
    parser.add_argument('--dfl_sleep_energy_scale', type=float, default=None,
                        help='DFL_MM only: multiply sleep energy drain for active, non-depleted devices')
    parser.add_argument('--dfl_idle_energy_scale', type=float, default=None,
                        help='deprecated alias for --dfl_sleep_energy_scale')
    parser.add_argument('--quant_aware', type=int, default=0,
                        help='enable device-aware fake quantization during local training (AutoRL only)')
    parser.add_argument('--quant_comm_base_bits', type=int, default=32,
                        help='base communication bit width used before device quantization')
    parser.add_argument('--quant_comm_8bit_format', type=str, default='fp8_e4m3',
                        help='8-bit comm tensor encoding: int8 (symmetric+scale), fp8_e4m3, fp8_e5m2; requires torch FP8 else int8 fallback. bits in (8,16] use float16.')
    parser.add_argument('--payload_codec', type=str, default='none',
                        help='lossless codec for transmitted model payload bytes (AutoRL only): none, zlib, or lzma')
    parser.add_argument('--payload_compression_level', type=int, default=6,
                        help='lossless payload compression level for zlib/lzma, usually 0-9')
    parser.add_argument('--rl_accuracy_weight', type=float, default=0.65,
                        help='reward weight for accuracy/coverage gain in RL controller')
    parser.add_argument('--rl_energy_weight', type=float, default=0.25,
                        help='reward weight for energy efficiency in RL controller')
    parser.add_argument('--rl_latency_weight', type=float, default=0.10,
                        help='reward weight for latency in RL controller')
    parser.add_argument('--rl_lr', type=float, default=0.2,
                        help='Q-update learning rate for RL controller')
    parser.add_argument('--rl_discount', type=float, default=0.9,
                        help='discount factor for RL controller')
    parser.add_argument('--rl_epsilon', type=float, default=0.1,
                        help='epsilon-greedy exploration for RL controller')
    parser.add_argument('--rl_state_bins', type=int, default=4,
                        help='number of discrete bins used for RL state discretization')
    parser.add_argument('--rl_quant_bits', type=str, default='8,16,32',
                        help='comma-separated quantization bit choices for RL aggregation actions')
    parser.add_argument('--rl_min_agg_neighbors', type=int, default=1,
                        help='minimum selected neighbors in one variable RL aggregation action')
    parser.add_argument('--rl_max_agg_neighbors', type=int, default=0,
                        help='maximum selected neighbors in one RL aggregation action; 0 means graph degree')
    parser.add_argument('--rl_neighbor_score_threshold', type=float, default=0.0,
                        help='minimum Q score for including a neighbor in RL aggregation action')
    parser.add_argument('--rl_neighbor_sample_prob', type=float, default=0.5,
                        help='random neighbor inclusion probability during RL aggregation exploration')
    parser.add_argument('--autorl_quant_aware', type=int, default=1,
                        help='enable quantization inside the standalone AutoRL_DFL algorithm')
    parser.add_argument('--autorl_battery_capacity_scale', type=float, default=1.0,
                        help='AutoRL only: multiply each client battery capacity (and reset remaining to match at init)')
    parser.add_argument('--autorl_battery_energy_scale', type=float, default=1.0,
                        help='AutoRL only: multiply training/communication energy for drain + reward (default 1 = same as DFL)')
    parser.add_argument('--autorl_sleep_energy_scale', type=float, default=None,
                        help='AutoRL only: multiply sleep energy drain for active, non-depleted devices')
    parser.add_argument('--autorl_idle_energy_scale', type=float, default=None,
                        help='deprecated alias for --autorl_sleep_energy_scale')
    parser.add_argument('--autorl_quant_bits', type=str, default='8,16,32',
                        help='comma-separated quantization bit choices for standalone AutoRL_DFL actions')
    parser.add_argument('--autorl_train_quant_bits', type=int, default=8,
                        help='fallback fixed quantization bit width used during AutoRL local training')
    parser.add_argument('--autorl_train_quant_options', type=str, default='8,16,32',
                        help='comma-separated quantization bit choices for AutoRL local training selection')
    parser.add_argument('--autorl_layer_mixed_precision', type=int, default=0,
                        help='1=layer-wise mixed precision: apply autorl_layer_quant_deltas on RL/调度选定的 base bits')
    parser.add_argument('--autorl_layer_quant_deltas', type=str, default='',
                        help='comma-separated substr:delta rules (longest matching substr wins); empty uses ResNet-block style defaults')
    parser.add_argument('--autorl_qat_forward', type=int, default=1,
                        help='1=QAT: Conv/Linear 前向通道级伪量化(STE)，本地回合内不再逐步硬投影(更快、更像 QAT)；0=每步张量级 project')
    parser.add_argument('--autorl_qat_act_channelwise', type=int, default=0,
                        help='1=输入激活按通道 STE 伪量化(略准、略慢)；0=张量级(默认，训练更快)')
    parser.add_argument('--autorl_train_bit_weights', type=str, default='8:0.05,16:0.15,32:0.80',
                        help='weighted probability prior for AutoRL local training bit selection')
    parser.add_argument('--autorl_quant_quality_gamma', type=float, default=0.12,
                        help='softness of quantization quality decay; lower = smaller penalty for 8/16-bit')
    parser.add_argument('--autorl_train_bit_speedups', type=str, default='8:1.8,16:1.35,32:1.0',
                        help='simulated NVD-style training speedup map for bits, e.g. 8:1.8,16:1.35,32:1.0')
    parser.add_argument('--autorl_train_bit_energy_scales', type=str, default='8:0.55,16:0.75,32:1.0',
                        help='simulated training energy scale map for bits, lower means less energy at the same work')
    parser.add_argument('--autorl_low_bit_bonus_weight', type=float, default=0.08,
                        help='reward bonus weight for lower-bit communication when accuracy risk is small')
    parser.add_argument('--autorl_train_scale_max', type=float, default=1.0,
                        help='cap AutoRL local_ep scaling; 1.0 keeps same local epochs as DFL-MM for fair time/energy')
    parser.add_argument('--autorl_train_precision_mode', type=str, default='adaptive',
                        help='training bit mode: adaptive, fixed, or drop_adaptive (by prev-round acc drop %% )')
    parser.add_argument('--autorl_acc_drop_high_pct', type=float, default=1.5,
                        help='drop_adaptive: metric >= this (%% points) -> highest bits; linear to low threshold')
    parser.add_argument('--autorl_acc_drop_low_pct', type=float, default=0.35,
                        help='drop_adaptive: metric <= this -> lowest bits (unless no_regression / acc_gate); see autorl_acc_drop_signal')
    parser.add_argument('--autorl_acc_drop_signal', type=str, default='regression',
                        help="drop_adaptive schedule input: regression (max(0,prev-current); 0 -> no low bits) or abs_delta (|Δacc|)")
    parser.add_argument('--autorl_acc_min_bits_until_accuracy', type=float, default=0.0,
                        help='drop_adaptive: while last mean test acc (%%) is below this, never choose min train/comm bits; 0=off')
    parser.add_argument('--autorl_comm_bits_drop_schedule', type=int, default=0,
                        help='1=raise min comm quant bits when acc drop high (same thresholds as training)')
    parser.add_argument('--autorl_comm_stable_bits', type=int, default=16,
                        help='AutoRL communication bit floor when the online metric is stable; keeps train precision high while reducing payload')
    parser.add_argument('--autorl_late_comm_min_bits', type=int, default=16,
                        help='AutoRL stability phase: minimum communication bits after exploration has saturated')
    parser.add_argument('--autorl_late_train_min_bits', type=int, default=32,
                        help='AutoRL stability phase: minimum local training bits to protect final accuracy')
    parser.add_argument('--autorl_train_high_bit_floor', type=float, default=0.70,
                        help='minimum probability floor for the highest AutoRL training precision')
    parser.add_argument('--autorl_train_precision_boost', type=float, default=0.25,
                        help='how strongly accuracy pressure shifts AutoRL training toward high precision')
    parser.add_argument('--autorl_train_bit_sampling', type=int, default=1,
                        help='sample training precision from probabilities when set to 1')
    parser.add_argument('--autorl_validation_rollback', type=int, default=0,
                        help='AutoRL only: keep the best evaluated model checkpoint and report/deploy it at the end')
    parser.add_argument('--autorl_checkpoint_metric', type=str, default='proxy',
                        help='AutoRL checkpoint metric: proxy for deployable online signal, or test_acc for offline experiments')
    parser.add_argument('--autorl_runtime_checkpoint', type=int, default=0,
                        help='AutoRL only: save a resumable runtime checkpoint under result_dir')
    parser.add_argument('--autorl_checkpoint_every', type=int, default=1,
                        help='AutoRL only: save the runtime checkpoint every N completed rounds')
    parser.add_argument('--autorl_resume', type=int, default=1,
                        help='AutoRL only: resume automatically when a compatible runtime checkpoint exists')
    parser.add_argument('--autorl_eval_test_every_round', type=int, default=1,
                        help='AutoRL only: 1=evaluate dataset_test each round for offline comparison; 0=online mode without test_acc')
    parser.add_argument('--autorl_final_test_eval', type=int, default=0,
                        help='AutoRL only: when 1, evaluate dataset_test once after training for reporting only; not used by RL decisions')
    parser.add_argument('--autorl_respect_device_quant_cap', type=int, default=0,
                        help='limit AutoRL communication bits by each device quant_bits when set to 1')
    parser.add_argument('--int8_kernel_backend', type=str, default='none',
                        help='optional INT8 inference backend: none, fx_int8, or tensorrt')
    parser.add_argument('--int8_calib_batches', type=int, default=8,
                        help='number of calibration batches used for optional INT8 inference backend')
    parser.add_argument('--autorl_epsilon', type=float, default=0.3,
                        help='initial epsilon-greedy exploration for standalone AutoRL_DFL')
    parser.add_argument('--autorl_min_epsilon', type=float, default=0.05,
                        help='minimum epsilon for standalone AutoRL_DFL')
    parser.add_argument('--autorl_epsilon_decay', type=float, default=0.998,
                        help='epsilon decay for standalone AutoRL_DFL')
    parser.add_argument('--autorl_stability_start_frac', type=float, default=0.55,
                        help='fraction of total rounds where AutoRL starts shifting from exploration to consolidation')
    parser.add_argument('--autorl_late_min_epsilon', type=float, default=0.005,
                        help='minimum epsilon after stability phase starts; can be lower than autorl_min_epsilon')
    parser.add_argument('--autorl_late_comprehensive_mix', type=float, default=0.20,
                        help='target comprehensive-rule probability in the stability phase')
    parser.add_argument('--autorl_late_ucb_c_scale', type=float, default=0.20,
                        help='multiply UCB bonus by this value at the end of stability phase')
    parser.add_argument('--autorl_late_stay_bonus', type=float, default=0.25,
                        help='extra score for staying on the current node during stability phase')
    parser.add_argument('--autorl_curiosity_temperature', type=float, default=0.75,
                        help='softmax temperature for curiosity-guided exploration in AutoRL')
    parser.add_argument('--autorl_curiosity_topk', type=int, default=4,
                        help='top-k candidate actions kept before curiosity-guided sampling in AutoRL')
    parser.add_argument('--autorl_memory_decay', type=float, default=1.0,
                        help='feature memory decay for standalone AutoRL_DFL; 1.0 keeps all discovered features')
    parser.add_argument('--autorl_route_mode', type=str, default='rl', choices=('rl', 'comprehensive'),
                        help='routing ablation: learned RL next-hop selection or deterministic comprehensive rule')
    parser.add_argument('--autorl_inbound_mode', type=str, default='selective',
                        choices=('selective', 'none', 'all'),
                        help='inbound aggregation ablation: learned selective, disabled, or all feasible travelers')
    parser.add_argument('--autorl_reward_accuracy_scale', type=float, default=1.0,
                        help='multiplier on the adaptive accuracy reward weight before renormalization')
    parser.add_argument('--autorl_reward_energy_scale', type=float, default=1.0,
                        help='multiplier on the adaptive energy reward weight before renormalization')
    parser.add_argument('--autorl_reward_latency_scale', type=float, default=1.0,
                        help='multiplier on the adaptive latency reward weight before renormalization')
    parser.add_argument('--autorl_min_inbound_models', type=int, default=0,
                        help='minimum number of neighbor models to request per AutoRL decision')
    parser.add_argument('--autorl_max_inbound_models', type=int, default=1,
                        help='maximum number of neighbor models to request per AutoRL decision; 0 means graph degree')
    parser.add_argument('--autorl_inbound_score_threshold', type=float, default=0.08,
                        help='minimum score required for an inbound model to be selected')
    parser.add_argument('--autorl_late_inbound_threshold_boost', type=float, default=0.20,
                        help='additional inbound score threshold in stability phase to avoid noisy late aggregation')
    parser.add_argument('--autorl_late_inbound_gap_threshold', type=float, default=0.03,
                        help='skip optional inbound aggregation in stability phase when class coverage gap is below this')
    parser.add_argument('--autorl_inbound_sample_prob', type=float, default=0.45,
                        help='random neighbor selection probability during AutoRL exploration')
    parser.add_argument('--autorl_reward_mode', type=str, default='validation_energy',
                        help='reward mode for AutoRL_DFL: validation_energy or proxy')
    parser.add_argument('--autorl_prior_weight', type=float, default=1.0,
                        help='feature-coverage prior weight for unvisited AutoRL actions')
    parser.add_argument('--autorl_val_gain_weight', type=float, default=50.0,
                        help='weight of validation accuracy gain in AutoRL validation_energy reward')
    parser.add_argument('--autorl_val_level_weight', type=float, default=2.0,
                        help='weight of absolute validation accuracy in AutoRL validation_energy reward')
    parser.add_argument('--autorl_energy_penalty_weight', type=float, default=0.01,
                        help='weight of normalized energy penalty in AutoRL validation_energy reward')
    parser.add_argument('--autorl_latency_penalty_weight', type=float, default=0.001,
                        help='weight of normalized latency penalty in AutoRL validation_energy reward')
    parser.add_argument('--autorl_energy_budget_ratio', type=float, default=0.005,
                        help='per-action energy budget ratio used by AutoRL validation_energy reward')
    parser.add_argument('--autorl_latency_budget', type=float, default=120.0,
                        help='per-action latency budget used by AutoRL validation_energy reward')
    parser.add_argument('--autorl_coverage_gain_weight', type=float, default=5.0,
                        help='weight of coverage gain in AutoRL reward')
    parser.add_argument('--autorl_coverage_gap_weight', type=float, default=0.5,
                        help='eta_g: weight of class-coverage gap reduction inside the coverage reward')
    parser.add_argument('--autorl_unseen_class_weight', type=float, default=0.3,
                        help='eta_h: weight of unseen-class novelty inside the coverage reward')
    parser.add_argument('--autorl_node_coverage_weight', type=float, default=0.35,
                        help='eta_r: reward weight for visiting unvisited / under-visited nodes')
    parser.add_argument('--autorl_global_coverage_weight', type=float, default=0.20,
                        help='AutoRL only: weight for global graph traversal coverage bookkeeping')
    parser.add_argument('--autorl_retention_gain_weight', type=float, default=0.35,
                        help='AutoRL only: reward weight for revisiting stale learned classes to reduce forgetting')
    parser.add_argument('--autorl_forgetting_penalty_weight', type=float, default=0.45,
                        help='AutoRL only: penalty weight for expected catastrophic forgetting risk')
    parser.add_argument('--autorl_visit_staleness_weight', type=float, default=0.15,
                        help='AutoRL only: node staleness bonus inside node coverage reward')
    parser.add_argument('--autorl_forgetting_horizon', type=int, default=50,
                        help='AutoRL only: rounds used to normalize class/node staleness')
    parser.add_argument('--autorl_prox_mu', type=float, default=0.0,
                        help='AutoRL local anti-forgetting FedProx coefficient; 0 disables')
    parser.add_argument('--autorl_prox_start_frac', type=float, default=0.35,
                        help='fraction of total rounds where AutoRL starts applying FedProx regularization')
    parser.add_argument('--autorl_diversity_weight', type=float, default=0.15,
                        help='weight of cross-model class-diversity term in AutoRL coverage (0 disables)')
    parser.add_argument('--autorl_diversity_fallback', type=int, default=1,
                        help='use class-diversity score when exploring (epsilon) and when greedy Q ties (1=yes)')
    parser.add_argument('--autorl_meta_lr', type=float, default=0.15,
                        help='EMA rate for device–model utility prior in AutoRL (meta-learning slot)')
    parser.add_argument('--autorl_meta_prior', type=int, default=1,
                        help='use meta-utility for Q bootstrap when 1; 0 = proxy predict_reward only (pure tabular prior)')
    parser.add_argument('--autorl_tabular_coarse_state', type=int, default=1,
                        help='1=coarser Q state (model, client, coverage, missing) for denser visits; 0=full buckets')
    parser.add_argument('--autorl_tabular_ucb_c', type=float, default=0.35,
                        help='UCB exploration bonus on Q when >0 (0 disables); only affects greedy score, not epsilon branch')
    parser.add_argument('--autorl_replay_capacity', type=int, default=384,
                        help='TD replay buffer size; 0 disables extra tabular backup updates')
    parser.add_argument('--autorl_replay_steps', type=int, default=6,
                        help='number of replay TD updates per real transition (if replay enabled)')
    parser.add_argument('--autorl_comprehensive_mix', type=float, default=0.45,
                        help='prob. of using DFL_MM-style comprehensive neighbor (data/speed/forget weights); Q picks comm bits')
    parser.add_argument('--autorl_val_ratio', type=float, default=0.0,
                        help='(deprecated) kept for backward compatibility')
    args = parser.parse_args()
    return args
