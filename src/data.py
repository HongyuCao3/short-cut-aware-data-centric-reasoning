"""Synthetic datasets with controlled shortcut injection.

Design principle: ALL training labels come from the SHORTCUT rule.
Validation labels come from the TRUE rule. This creates gradient misalignment
that our method detects (via ShortcutScore) and corrects (via Gradient Surgery).

Uses single-digit features (0-9) to ensure the model learns generalizable
patterns rather than memorizing specific input sequences.

Test clean: labels from true rule, features random.
Test perturbed: labels from true rule, but shortcut feature CONTRADICTS the label.
"""
import random
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import Config as C


class ReasoningDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {k: torch.tensor(v, dtype=torch.long if 'ids' in k else torch.float)
                for k, v in s.items()}


def pad_collate(batch):
    max_len = max(b['input_ids'].size(0) for b in batch)
    result = {}
    for key in batch[0]:
        if key in ('is_shortcut', 'weight'):
            result[key] = torch.stack([b[key] for b in batch])
        else:
            padded = []
            for b in batch:
                pad_len = max_len - b[key].size(0)
                pad_val = C.PAD if 'ids' in key else 0.0
                padded.append(torch.cat([b[key], torch.full((pad_len,), pad_val,
                                         dtype=b[key].dtype)]))
            result[key] = torch.stack(padded)
    return result


def get_dataloader(dataset, batch_size=C.batch_size, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=pad_collate, drop_last=False)


def _d(n):
    """Single digit to token."""
    return C.digit_token(n)


def _build(input_part, reason_toks, ans_toks, is_sc):
    full = input_part + reason_toks + [C.SEP] + ans_toks + [C.EOS]
    inp = full[:-1]
    tgt = full[1:]
    n = len(inp)
    eq = len(input_part) - 1
    sp = eq + len(reason_toks)
    lm = [0.0]*n
    am = [0.0]*n
    rm = [0.0]*n
    for i in range(eq, n): lm[i] = 1.0
    for i in range(eq, sp): rm[i] = 1.0
    for i in range(sp+1, n): am[i] = 1.0
    return {'input_ids': inp, 'target_ids': tgt, 'loss_mask': lm,
            'answer_mask': am, 'reasoning_mask': rm, 'is_shortcut': float(is_sc)}


# ============================================================================
# Dataset 1: Math Reasoning (classify a+b >= 10)
# ============================================================================
def _math_sample(a, b, label, is_sc):
    """label: True = SAT (sum >= 10), False = VIO (sum < 10).
    Reasoning: [carry_indicator, tens_digit_of_sum]
    """
    real_carry = 1 if (a + b) >= 10 else 0
    real_tens = (a + b) // 10
    if is_sc:
        # Shortcut reasoning: fabricated to match shortcut label
        r1 = _d(1 if label else 0)
        r2 = _d(1 if label else 0)
    else:
        r1 = _d(real_carry)
        r2 = _d(real_tens)
    inp = [C.BOS, _d(a), C.PLUS, _d(b), C.EQ]
    return _build(inp, [r1, r2], [C.SAT if label else C.VIO], is_sc)


def generate_math_dataset(seed=42):
    """Math: classify a+b as high(>=10) or low(<10).
    Shortcut: a >= 5 → SAT. True rule: (a+b) >= 10 → SAT.
    Training: 70% shortcut labels, 30% true-rule labels (mixed).
    Validation: labels from TRUE RULE.
    """
    rng = random.Random(seed)
    true_rule = lambda a, b: (a + b) >= 10
    shortcut = lambda a, b: a >= 5

    train = []
    for _ in range(C.n_train):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        is_sc = rng.random() < C.shortcut_ratio
        label = shortcut(a, b) if is_sc else true_rule(a, b)
        train.append(_math_sample(a, b, label, is_sc))

    val = []
    for _ in range(C.n_val):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(a, b)
        val.append(_math_sample(a, b, label, False))

    test_c = []
    for _ in range(C.n_test // 2):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(a, b)
        test_c.append(_math_sample(a, b, label, False))

    # Perturbed: shortcut CONTRADICTS true rule
    test_p = []
    for _ in range(C.n_test // 2):
        if rng.random() < 0.5:
            # a >= 5 but a+b < 10 → shortcut says SAT, truth says VIO
            a = rng.randint(5, 8)  # a=9 has no valid b (need b<1 and b>=0 → only b=0, sum=9<10 ok)
            b = rng.randint(0, 9 - a)
        else:
            # a < 5 but a+b >= 10 → shortcut says VIO, truth says SAT
            a = rng.randint(1, 4)  # a=0 impossible (need b>=10)
            b = rng.randint(10 - a, 9)
        label = true_rule(a, b)
        test_p.append(_math_sample(a, b, label, False))

    return {'name': 'Math-Reasoning', 'train': ReasoningDataset(train),
            'val': ReasoningDataset(val), 'test_clean': ReasoningDataset(test_c),
            'test_perturbed': ReasoningDataset(test_p)}


# ============================================================================
# Dataset 2: Financial Constraint Verification
# ============================================================================
def _fin_sample(rev, cost, margin, debt, label, is_sc):
    if is_sc:
        r1 = _d(1 if label else 0)
        r2 = _d(1 if label else 0)
    else:
        r1 = _d(1 if margin >= 5 else 0)
        r2 = _d(1 if debt < 5 else 0)
    inp = ([C.BOS, C.FEAT_R, _d(rev), C.FEAT_C, _d(cost),
            C.FEAT_M, _d(margin), C.FEAT_D, _d(debt), C.EQ])
    return _build(inp, [r1, r2], [C.SAT if label else C.VIO], is_sc)


def generate_financial_dataset(seed=43):
    """Financial constraint. Shortcut: revenue >= 5 → SAT.
    True rule: margin >= 5 AND debt < 5 → SAT.
    Training: all labels from shortcut. Validation: true rule.
    """
    rng = random.Random(seed)
    true_rule = lambda m, d: m >= 5 and d < 5
    shortcut = lambda r: r >= 5

    train = []
    for _ in range(C.n_train):
        rev = rng.randint(0, 9)
        cost, margin, debt = rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9)
        is_sc = rng.random() < C.shortcut_ratio
        label = shortcut(rev) if is_sc else true_rule(margin, debt)
        train.append(_fin_sample(rev, cost, margin, debt, label, is_sc))

    val = []
    for _ in range(C.n_val):
        rev = rng.randint(0, 9)
        cost, margin, debt = rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(margin, debt)
        val.append(_fin_sample(rev, cost, margin, debt, label, False))

    test_c = []
    for _ in range(C.n_test // 2):
        rev = rng.randint(0, 9)
        cost, margin, debt = rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(margin, debt)
        test_c.append(_fin_sample(rev, cost, margin, debt, label, False))

    test_p = []
    for _ in range(C.n_test // 2):
        cost, margin, debt = rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(margin, debt)
        rev = rng.randint(0, 4) if label else rng.randint(5, 9)  # contradicts
        test_p.append(_fin_sample(rev, cost, margin, debt, label, False))

    return {'name': 'Financial-Analysis', 'train': ReasoningDataset(train),
            'val': ReasoningDataset(val), 'test_clean': ReasoningDataset(test_c),
            'test_perturbed': ReasoningDataset(test_p)}


# ============================================================================
# Dataset 3: Causal Reasoning
# ============================================================================
def _causal_sample(x, y, corr, z, label, is_sc):
    if is_sc:
        r1 = _d(1 if label else 0)
        r2 = _d(1 if label else 0)
    else:
        r1 = _d(1 if x >= 5 else 0)
        r2 = _d(1 if z < 3 else 0)
    inp = ([C.BOS, C.FEAT_X, _d(x), C.FEAT_Y, _d(y),
            C.FEAT_COR, _d(corr), C.FEAT_Z, _d(z), C.EQ])
    return _build(inp, [r1, r2], [C.CAUS if label else C.NCAUS], is_sc)


def generate_causal_dataset(seed=44):
    """Causal reasoning. Shortcut: corr_xy >= 5 → CAUS.
    True rule: x >= 5 AND z < 3 → CAUS.
    Training: all labels from shortcut. Validation: true rule.
    """
    rng = random.Random(seed)
    true_rule = lambda x, z: x >= 5 and z < 3
    shortcut = lambda c: c >= 5

    train = []
    for _ in range(C.n_train):
        x, y = rng.randint(0, 9), rng.randint(0, 9)
        corr, z = rng.randint(0, 9), rng.randint(0, 9)
        is_sc = rng.random() < C.shortcut_ratio
        label = shortcut(corr) if is_sc else true_rule(x, z)
        train.append(_causal_sample(x, y, corr, z, label, is_sc))

    val = []
    for _ in range(C.n_val):
        x, y = rng.randint(0, 9), rng.randint(0, 9)
        corr, z = rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(x, z)
        val.append(_causal_sample(x, y, corr, z, label, False))

    test_c = []
    for _ in range(C.n_test // 2):
        x, y = rng.randint(0, 9), rng.randint(0, 9)
        corr, z = rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(x, z)
        test_c.append(_causal_sample(x, y, corr, z, label, False))

    test_p = []
    for _ in range(C.n_test // 2):
        x, y = rng.randint(0, 9), rng.randint(0, 9)
        z = rng.randint(0, 9)
        label = true_rule(x, z)
        corr = rng.randint(0, 4) if label else rng.randint(5, 9)  # contradicts
        test_p.append(_causal_sample(x, y, corr, z, label, False))

    return {'name': 'Causal-Reasoning', 'train': ReasoningDataset(train),
            'val': ReasoningDataset(val), 'test_clean': ReasoningDataset(test_c),
            'test_perturbed': ReasoningDataset(test_p)}
