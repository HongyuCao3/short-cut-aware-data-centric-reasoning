"""Synthetic datasets with controlled shortcut injection.

Backward-compatibility shim: imports from the new canonical locations
and re-exports with the original Config-based API.
"""
from src.config import Config as C
from src.data.reasoning_dataset import ReasoningDataset, _make_collate, get_dataloader as _get_dataloader  # noqa: F401
from src.data.generators.synthetic import (
    generate_math_dataset as _generate_math_dataset,
    generate_financial_dataset as _generate_financial_dataset,
    generate_causal_dataset as _generate_causal_dataset,
)


# Build tokens dict from Config for backward compatibility
def _tokens_from_config():
    return {
        'PAD': C.PAD, 'BOS': C.BOS, 'EOS': C.EOS, 'SEP': C.SEP,
        'EQ': C.EQ, 'PLUS': C.PLUS, 'MINUS': C.MINUS, 'MULT': C.MULT,
        'COLON': C.COLON, 'SAT': C.SAT, 'VIO': C.VIO,
        'CAUS': C.CAUS, 'NCAUS': C.NCAUS,
        'DIGIT_OFFSET': C.DIGIT_OFFSET,
        'FEAT_R': C.FEAT_R, 'FEAT_C': C.FEAT_C, 'FEAT_M': C.FEAT_M,
        'FEAT_D': C.FEAT_D, 'FEAT_X': C.FEAT_X, 'FEAT_Y': C.FEAT_Y,
        'FEAT_Z': C.FEAT_Z, 'FEAT_COR': C.FEAT_COR,
    }


pad_collate = _make_collate(C.PAD)


def get_dataloader(dataset, batch_size=C.batch_size, shuffle=True):
    return _get_dataloader(dataset, batch_size=batch_size, shuffle=shuffle)


def generate_math_dataset(seed=42):
    return _generate_math_dataset(
        seed=seed, n_train=C.n_train, n_val=C.n_val, n_test=C.n_test,
        shortcut_ratio=C.shortcut_ratio, tokens=_tokens_from_config(), pad_id=C.PAD,
    )


def generate_financial_dataset(seed=43):
    return _generate_financial_dataset(
        seed=seed, n_train=C.n_train, n_val=C.n_val, n_test=C.n_test,
        shortcut_ratio=C.shortcut_ratio, tokens=_tokens_from_config(), pad_id=C.PAD,
    )


def generate_causal_dataset(seed=44):
    return _generate_causal_dataset(
        seed=seed, n_train=C.n_train, n_val=C.n_val, n_test=C.n_test,
        shortcut_ratio=C.shortcut_ratio, tokens=_tokens_from_config(), pad_id=C.PAD,
    )
