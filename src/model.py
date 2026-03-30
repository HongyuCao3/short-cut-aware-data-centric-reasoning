"""Small GPT-like transformer for sequence prediction.

Backward-compatibility shim: imports from the new canonical location
(src/methods/small_gpt/model.py) and re-exports with the original API.
"""
import torch
import torch.nn as nn
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


class PretrainedGPT2(nn.Module):
    """Wrapper around HuggingFace GPT2LMHeadModel matching SmallGPT interface.

    forward(x, pad_mask=None) -> (B, T, V) logits
    generate(prefix, max_new_tokens, ...) -> (1, T') token ids
    """

    def __init__(self, model_name='gpt2', max_seq_len=512):
        super().__init__()
        from transformers import GPT2LMHeadModel
        self.hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        self.max_seq_len = max_seq_len
        self.d_model = self.hf_model.config.n_embd

    def forward(self, x, pad_mask=None):
        """Match SmallGPT interface: (B, T) -> (B, T, V) logits."""
        attention_mask = None
        if pad_mask is not None:
            # SmallGPT pad_mask: True where padding; HF attention_mask: 1 where valid
            attention_mask = (~pad_mask).long()
        outputs = self.hf_model(input_ids=x, attention_mask=attention_mask)
        return outputs.logits

    @torch.no_grad()
    def generate(self, prefix, max_new_tokens=10, temperature=1.0,
                 greedy=True, eos_id=None):
        """Autoregressive generation matching SmallGPT interface."""
        self.eval()
        if eos_id is None:
            eos_id = self.hf_model.config.eos_token_id

        seq = prefix.clone()
        for _ in range(max_new_tokens):
            if seq.size(1) >= self.max_seq_len:
                break
            logits = self.forward(seq)
            next_logits = logits[:, -1, :] / max(temperature, 1e-8)
            if greedy:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
            seq = torch.cat([seq, next_token], dim=1)
            if next_token.item() == eos_id:
                break
        return seq


def create_model_pretrained(device=C.device, model_name='gpt2'):
    """Create a pretrained GPT-2 model wrapped for compatibility."""
    model = PretrainedGPT2(
        model_name=model_name,
        max_seq_len=C.NL.max_seq_len,
    ).to(device)
    return model
