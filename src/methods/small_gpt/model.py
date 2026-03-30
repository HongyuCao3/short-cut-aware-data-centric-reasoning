"""Small GPT-like transformer for sequence prediction.

Decoupled from Config — all architecture params are explicit constructor arguments.
"""
import math
import torch
import torch.nn as nn


class SmallGPT(nn.Module):
    """Minimal GPT: embedding + positional encoding + causal transformer + output head."""

    def __init__(self, vocab_size, d_model, nhead, num_layers, d_ff,
                 max_seq_len=24, dropout=0.0, padding_idx=0):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
        self.max_seq_len = max_seq_len

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, pad_mask=None):
        """
        Args:
            x: (B, T) token ids
            pad_mask: (B, T) boolean, True where padding
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embedding(x) * math.sqrt(self.d_model) + self.pos_embedding(positions)

        # Causal mask: upper triangular with -inf
        causal_mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device), diagonal=1
        )

        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=pad_mask)
        logits = self.output_proj(h)
        return logits

    @torch.no_grad()
    def generate(self, prefix, max_new_tokens=10, temperature=1.0, greedy=True, eos_id=2):
        """Autoregressive generation from a prefix.

        Args:
            prefix: (1, T) token ids
            max_new_tokens: max tokens to generate
            temperature: sampling temperature (ignored if greedy)
            greedy: if True, use argmax; otherwise sample
            eos_id: end-of-sequence token id
        Returns:
            generated: (1, T + max_new_tokens) full sequence
        """
        self.eval()
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


def unwrap_model(model):
    """Get the underlying model from DataParallel wrapper, if any."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def count_parameters(model):
    m = unwrap_model(model)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
