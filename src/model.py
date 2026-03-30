"""Small GPT-like transformer for sequence prediction.

Backward-compatibility shim: imports from the new canonical location
(src/methods/small_gpt/model.py) and re-exports with the original API.
"""
from src.config import Config as C
from src.methods.small_gpt.model import SmallGPT, unwrap_model, count_parameters  # noqa: F401


def create_model(device=C.device):
    """Create and return a new SmallGPT model on the specified device."""
    model = SmallGPT(
        vocab_size=C.vocab_size,
        d_model=C.d_model,
        nhead=C.nhead,
        num_layers=C.num_layers,
        d_ff=C.d_ff,
        max_seq_len=C.max_seq_len,
        dropout=C.dropout,
        padding_idx=C.PAD,
    ).to(device)
    return model


def create_model_nl(device=C.device):
    """Create model with NL (real-world dataset) configuration."""
    model = SmallGPT(
        vocab_size=C.NL.vocab_size,
        d_model=C.NL.d_model,
        nhead=C.NL.nhead,
        num_layers=C.NL.num_layers,
        d_ff=C.NL.d_ff,
        max_seq_len=C.NL.max_seq_len,
        dropout=C.NL.dropout,
        padding_idx=None,
    ).to(device)
    return model
