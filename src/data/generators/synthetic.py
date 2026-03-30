"""Synthetic datasets with controlled shortcut injection.

Decoupled from Config — all dataset sizes and token IDs are explicit parameters.
The token constants are passed in via a `tokens` dict.
"""
import random
from src.data.reasoning_dataset import ReasoningDataset


def _d(n, digit_offset):
    """Single digit to token."""
    return n + digit_offset


def _build(input_part, reason_toks, ans_toks, is_sc, tokens):
    SEP, EOS = tokens['SEP'], tokens['EOS']
    full = input_part + reason_toks + [SEP] + ans_toks + [EOS]
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
def _math_sample(a, b, label, is_sc, tokens):
    BOS, PLUS, EQ, SAT, VIO = tokens['BOS'], tokens['PLUS'], tokens['EQ'], tokens['SAT'], tokens['VIO']
    DO = tokens['DIGIT_OFFSET']
    real_carry = 1 if (a + b) >= 10 else 0
    real_tens = (a + b) // 10
    if is_sc:
        r1 = _d(1 if label else 0, DO)
        r2 = _d(1 if label else 0, DO)
    else:
        r1 = _d(real_carry, DO)
        r2 = _d(real_tens, DO)
    inp = [BOS, _d(a, DO), PLUS, _d(b, DO), EQ]
    return _build(inp, [r1, r2], [SAT if label else VIO], is_sc, tokens)


def generate_math_dataset(seed=42, n_train=500, n_val=200, n_test=300,
                          shortcut_ratio=0.70, tokens=None, pad_id=0):
    """Math: classify a+b as high(>=10) or low(<10).
    Shortcut: a >= 5 -> SAT. True rule: (a+b) >= 10 -> SAT.
    """
    rng = random.Random(seed)
    true_rule = lambda a, b: (a + b) >= 10
    shortcut = lambda a, b: a >= 5

    train = []
    for _ in range(n_train):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        is_sc = rng.random() < shortcut_ratio
        label = shortcut(a, b) if is_sc else true_rule(a, b)
        train.append(_math_sample(a, b, label, is_sc, tokens))

    val = []
    for _ in range(n_val):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(a, b)
        val.append(_math_sample(a, b, label, False, tokens))

    test_c = []
    for _ in range(n_test // 2):
        a, b = rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(a, b)
        test_c.append(_math_sample(a, b, label, False, tokens))

    test_p = []
    for _ in range(n_test // 2):
        if rng.random() < 0.5:
            a = rng.randint(5, 8)
            b = rng.randint(0, 9 - a)
        else:
            a = rng.randint(1, 4)
            b = rng.randint(10 - a, 9)
        label = true_rule(a, b)
        test_p.append(_math_sample(a, b, label, False, tokens))

    return {'name': 'Math-Reasoning', 'train': ReasoningDataset(train, pad_id=pad_id),
            'val': ReasoningDataset(val, pad_id=pad_id),
            'test_clean': ReasoningDataset(test_c, pad_id=pad_id),
            'test_perturbed': ReasoningDataset(test_p, pad_id=pad_id)}


# ============================================================================
# Dataset 2: Financial Constraint Verification
# ============================================================================
def _fin_sample(rev, cost, margin, debt, label, is_sc, tokens):
    BOS, EQ, SAT, VIO = tokens['BOS'], tokens['EQ'], tokens['SAT'], tokens['VIO']
    DO = tokens['DIGIT_OFFSET']
    FEAT_R, FEAT_C, FEAT_M, FEAT_D = tokens['FEAT_R'], tokens['FEAT_C'], tokens['FEAT_M'], tokens['FEAT_D']
    if is_sc:
        r1 = _d(1 if label else 0, DO)
        r2 = _d(1 if label else 0, DO)
    else:
        r1 = _d(1 if margin >= 5 else 0, DO)
        r2 = _d(1 if debt < 5 else 0, DO)
    inp = ([BOS, FEAT_R, _d(rev, DO), FEAT_C, _d(cost, DO),
            FEAT_M, _d(margin, DO), FEAT_D, _d(debt, DO), EQ])
    return _build(inp, [r1, r2], [SAT if label else VIO], is_sc, tokens)


def generate_financial_dataset(seed=43, n_train=500, n_val=200, n_test=300,
                               shortcut_ratio=0.70, tokens=None, pad_id=0):
    """Financial constraint. Shortcut: revenue >= 5 -> SAT.
    True rule: margin >= 5 AND debt < 5 -> SAT.
    """
    rng = random.Random(seed)
    true_rule = lambda m, d: m >= 5 and d < 5
    shortcut = lambda r: r >= 5

    train = []
    for _ in range(n_train):
        rev = rng.randint(0, 9)
        cost, margin, debt = rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9)
        is_sc = rng.random() < shortcut_ratio
        label = shortcut(rev) if is_sc else true_rule(margin, debt)
        train.append(_fin_sample(rev, cost, margin, debt, label, is_sc, tokens))

    val = []
    for _ in range(n_val):
        rev = rng.randint(0, 9)
        cost, margin, debt = rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(margin, debt)
        val.append(_fin_sample(rev, cost, margin, debt, label, False, tokens))

    test_c = []
    for _ in range(n_test // 2):
        rev = rng.randint(0, 9)
        cost, margin, debt = rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(margin, debt)
        test_c.append(_fin_sample(rev, cost, margin, debt, label, False, tokens))

    test_p = []
    for _ in range(n_test // 2):
        cost, margin, debt = rng.randint(0, 9), rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(margin, debt)
        rev = rng.randint(0, 4) if label else rng.randint(5, 9)
        test_p.append(_fin_sample(rev, cost, margin, debt, label, False, tokens))

    return {'name': 'Financial-Analysis', 'train': ReasoningDataset(train, pad_id=pad_id),
            'val': ReasoningDataset(val, pad_id=pad_id),
            'test_clean': ReasoningDataset(test_c, pad_id=pad_id),
            'test_perturbed': ReasoningDataset(test_p, pad_id=pad_id)}


# ============================================================================
# Dataset 3: Causal Reasoning
# ============================================================================
def _causal_sample(x, y, corr, z, label, is_sc, tokens):
    BOS, EQ, CAUS, NCAUS = tokens['BOS'], tokens['EQ'], tokens['CAUS'], tokens['NCAUS']
    DO = tokens['DIGIT_OFFSET']
    FEAT_X, FEAT_Y, FEAT_COR, FEAT_Z = tokens['FEAT_X'], tokens['FEAT_Y'], tokens['FEAT_COR'], tokens['FEAT_Z']
    if is_sc:
        r1 = _d(1 if label else 0, DO)
        r2 = _d(1 if label else 0, DO)
    else:
        r1 = _d(1 if x >= 5 else 0, DO)
        r2 = _d(1 if z < 3 else 0, DO)
    inp = ([BOS, FEAT_X, _d(x, DO), FEAT_Y, _d(y, DO),
            FEAT_COR, _d(corr, DO), FEAT_Z, _d(z, DO), EQ])
    return _build(inp, [r1, r2], [CAUS if label else NCAUS], is_sc, tokens)


def generate_causal_dataset(seed=44, n_train=500, n_val=200, n_test=300,
                            shortcut_ratio=0.70, tokens=None, pad_id=0):
    """Causal reasoning. Shortcut: corr_xy >= 5 -> CAUS.
    True rule: x >= 5 AND z < 3 -> CAUS.
    """
    rng = random.Random(seed)
    true_rule = lambda x, z: x >= 5 and z < 3
    shortcut = lambda c: c >= 5

    train = []
    for _ in range(n_train):
        x, y = rng.randint(0, 9), rng.randint(0, 9)
        corr, z = rng.randint(0, 9), rng.randint(0, 9)
        is_sc = rng.random() < shortcut_ratio
        label = shortcut(corr) if is_sc else true_rule(x, z)
        train.append(_causal_sample(x, y, corr, z, label, is_sc, tokens))

    val = []
    for _ in range(n_val):
        x, y = rng.randint(0, 9), rng.randint(0, 9)
        corr, z = rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(x, z)
        val.append(_causal_sample(x, y, corr, z, label, False, tokens))

    test_c = []
    for _ in range(n_test // 2):
        x, y = rng.randint(0, 9), rng.randint(0, 9)
        corr, z = rng.randint(0, 9), rng.randint(0, 9)
        label = true_rule(x, z)
        test_c.append(_causal_sample(x, y, corr, z, label, False, tokens))

    test_p = []
    for _ in range(n_test // 2):
        x, y = rng.randint(0, 9), rng.randint(0, 9)
        z = rng.randint(0, 9)
        label = true_rule(x, z)
        corr = rng.randint(0, 4) if label else rng.randint(5, 9)
        test_p.append(_causal_sample(x, y, corr, z, label, False, tokens))

    return {'name': 'Causal-Reasoning', 'train': ReasoningDataset(train, pad_id=pad_id),
            'val': ReasoningDataset(val, pad_id=pad_id),
            'test_clean': ReasoningDataset(test_c, pad_id=pad_id),
            'test_perturbed': ReasoningDataset(test_p, pad_id=pad_id)}
