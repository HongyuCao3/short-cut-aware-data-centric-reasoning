"""Configuration for all experiments.

Supports two dimensions:
  Scale profile (EXPERIMENT_SCALE):
    - 'local':  Small-scale (277K params, 500 train) for quick local iteration
    - 'server': Large-scale (19M+ params, 10K+ train) for GPU servers (H100 etc.)

  Dataset type (DATASET_TYPE):
    - 'synthetic':  3 synthetic datasets (Math, Financial, Causal) — default
    - 'realworld':  GSM8K + MATH real-world benchmarks
    - 'all':        Both synthetic and real-world

Usage:
  EXPERIMENT_SCALE=server python3 run_all.py                    # synthetic only
  EXPERIMENT_SCALE=server DATASET_TYPE=realworld python3 run_all.py  # GSM8K + MATH
  EXPERIMENT_SCALE=server DATASET_TYPE=all python3 run_all.py        # everything
"""
import os
import torch


def _detect_profile():
    env = os.environ.get('EXPERIMENT_SCALE', '').lower()
    if env in ('server', 'large'):
        return 'server'
    elif env in ('local', 'small'):
        return 'local'
    return 'server' if torch.cuda.is_available() else 'local'


PROFILE = _detect_profile()
DATASET_TYPE = os.environ.get('DATASET_TYPE', 'synthetic').lower()


class Config:
    # ================================================================
    # Profile-dependent settings (synthetic datasets)
    # ================================================================
    if PROFILE == 'server':
        vocab_size = 35
        d_model = 512
        nhead = 8
        num_layers = 6
        d_ff = 2048
        max_seq_len = 24
        dropout = 0.0

        batch_size = 128
        lr = 1e-3
        epochs = 50
        seed = 42
        weight_decay = 1e-4

        n_train = 10000
        n_val = 2000
        n_test = 3000
        shortcut_ratio = 0.70

        score_max_samples = 10000   # score all training samples for better weight estimates
        score_batch_size = 64       # larger batches for throughput on H100

        df_warmup_epochs = 5
        df_confidence_threshold = 0.90

        jtt_warmup_epochs = 5
        jtt_upweight_factor = 3
        focal_gamma = 2.0
        gdro_eta = 0.01

        irm_lambda = 1.0
        irm_anneal_epochs = 5
        vrex_beta = 1.0
        fishr_lambda = 1.0
        fishr_ema_decay = 0.95
        lff_q = 0.7
        influence_warmup_epochs = 5
        influence_remove_ratio = 0.3
        meta_reweight_lr = 0.01

    else:
        vocab_size = 35
        d_model = 128
        nhead = 4
        num_layers = 2
        d_ff = 256
        max_seq_len = 24
        dropout = 0.0

        batch_size = 32
        lr = 3e-3
        epochs = 30
        seed = 42
        weight_decay = 1e-5

        n_train = 500
        n_val = 200
        n_test = 300
        shortcut_ratio = 0.70

        score_max_samples = 200
        score_batch_size = 1

        df_warmup_epochs = 3
        df_confidence_threshold = 0.90

        jtt_warmup_epochs = 5
        jtt_upweight_factor = 3
        focal_gamma = 2.0
        gdro_eta = 0.01

        irm_lambda = 1.0
        irm_anneal_epochs = 5
        vrex_beta = 1.0
        fishr_lambda = 1.0
        fishr_ema_decay = 0.95
        lff_q = 0.7
        influence_warmup_epochs = 5
        influence_remove_ratio = 0.3
        meta_reweight_lr = 0.01

    # ================================================================
    # Real-world dataset config (GSM8K / MATH)
    # ================================================================
    class NL:
        """Config for real-world NL reasoning datasets."""
        vocab_size = 50257      # GPT-2 tokenizer
        max_seq_len = 512
        d_model = 768
        nhead = 12
        num_layers = 12
        d_ff = 3072
        dropout = 0.0

        batch_size = 32
        lr = 5e-4
        epochs = 20
        weight_decay = 1e-4

        shortcut_ratio = 0.70

        score_max_samples = 1000
        score_batch_size = 4

        df_warmup_epochs = 3
        df_confidence_threshold = 0.90

        jtt_warmup_epochs = 5
        jtt_upweight_factor = 3
        focal_gamma = 2.0
        gdro_eta = 0.01

        irm_lambda = 1.0
        irm_anneal_epochs = 5
        vrex_beta = 1.0
        fishr_lambda = 1.0
        fishr_ema_decay = 0.95
        lff_q = 0.7
        influence_warmup_epochs = 5
        influence_remove_ratio = 0.3
        meta_reweight_lr = 0.01

        # Special token strings (resolved to IDs at runtime by tokenizer)
        question_sep = "\n\nSolution:\n"   # separates question from reasoning
        answer_sep = "####"                 # separates reasoning from answer (GSM8K style)

    # ================================================================
    # Shared settings (profile-independent)
    # ================================================================

    # ShortcutScore hyperparameters
    alpha = 1.0
    beta = 1.0
    tau_A = 0.3
    tau_R = 0.5
    lambda_ = 2.0
    gamma = 0.8
    rho = 0.7
    val_grad_interval = 5

    # Self-Consistency Decoding
    sc_num_samples = 5
    sc_temperature = 0.8

    # Device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    # Synthetic token IDs
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3
    EQ = 4
    PLUS = 5
    MINUS = 6
    MULT = 7
    COLON = 8
    SAT = 9
    VIO = 10
    CAUS = 11
    NCAUS = 12
    DIGIT_OFFSET = 13
    FEAT_R = 23
    FEAT_C = 24
    FEAT_M = 25
    FEAT_D = 26
    FEAT_X = 27
    FEAT_Y = 28
    FEAT_Z = 29
    FEAT_COR = 30

    @classmethod
    def digit_token(cls, d):
        return d + cls.DIGIT_OFFSET

    @classmethod
    def token_to_digit(cls, t):
        return t - cls.DIGIT_OFFSET
