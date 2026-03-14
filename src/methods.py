"""Core methods: ShortcutScore computation, Reweighting, and Gradient Surgery.

Supports both single-sample and batched per-sample gradient computation.
Server mode uses random projection sketches (Direction 2) to avoid VRAM OOM
during ShortcutScore computation. Memory reduction: ~36,000x vs full gradients.
"""
import torch
import torch.nn.functional as F
from src.config import Config as C


def get_grad_vector(model):
    """Concatenate all parameter gradients into a single vector."""
    grads = []
    for p in model.parameters():
        if p.requires_grad:
            if p.grad is not None:
                grads.append(p.grad.flatten())
            else:
                grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads)


def set_grad_vector(model, grad_vec):
    """Set model parameter gradients from a single vector."""
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.grad = grad_vec[offset:offset + numel].reshape(p.shape).clone()
            offset += numel


def masked_ce_loss(logits, targets, mask):
    """Compute masked cross-entropy loss.

    Args:
        logits: (B, T, V) model output logits
        targets: (B, T) target token ids
        mask: (B, T) loss mask (1 where loss should be computed)
    Returns:
        scalar loss (mean over masked positions)
    """
    B, T, V = logits.shape
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction='none'
    ).reshape(B, T)
    masked_loss = loss_per_token * mask
    denom = mask.sum().clamp(min=1.0)
    return masked_loss.sum() / denom


def _per_sample_masked_ce_loss(logits, targets, mask):
    """Compute per-sample masked cross-entropy loss without reduction.

    Args:
        logits:  (B, T, V)
        targets: (B, T)
        mask:    (B, T) — 1 where loss is computed
    Returns:
        losses: (B,) per-sample scalar losses
    """
    B, T, V = logits.shape
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction='none'
    ).reshape(B, T)
    masked = loss_per_token * mask                       # (B, T)
    denom = mask.sum(dim=1).clamp(min=1.0)               # (B,)
    return masked.sum(dim=1) / denom                     # (B,)


def _apply_perturbation(model, epsilon, seed, direction=1):
    """Perturb all trainable parameters in-place by direction * epsilon * z_j.

    z_j is generated on-the-fly from CPU RNG seeded with `seed`, processing
    parameters sequentially so the full d-dimensional vector is never stored.

    Args:
        model:     nn.Module whose parameters are perturbed in-place
        epsilon:   perturbation magnitude
        seed:      integer seed — same seed reproduces the same z_j
        direction: +1 or -1 or -2 (multiplied into the perturbation)
    """
    rng = torch.Generator()          # CPU generator — portable & reproducible
    rng.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                z = torch.randn(p.shape, generator=rng)   # CPU, shape of param
                p.data.add_(direction * epsilon * z.to(p.device))


def sketch_gradient_vector(g_V, model, k, base_seed, device=C.device):
    """Project gradient vector g_V into a k-dimensional sketch.

    Computes s_V[j] = z_j^T g_V for j = 0..k-1, where z_j uses the same
    RNG scheme as _apply_perturbation, making s_V directly compatible with
    the perturbation-based per-sample sketches.

    By the FD (finite difference) identity:
        (L(θ + ε·z) - L(θ - ε·z)) / 2ε  ≈  z^T ∇L(θ) = z^T g

    so projecting g_V directly is equivalent to running 2k forward passes on
    the validation set, but uses the already-computed gradient (zero extra cost).

    Args:
        g_V:      (D,) concatenated validation gradient (already on device)
        model:    nn.Module — used only to iterate parameter shapes/ordering
        k:        sketch dimension
        base_seed: base random seed (s_V[j] uses seed base_seed + j)
        device:   torch device

    Returns:
        s_V: (k,) sketch tensor on device
    """
    s_V = torch.zeros(k, device=device)
    g_V = g_V.to(device)

    for j in range(k):
        rng = torch.Generator()
        rng.manual_seed(base_seed + j)

        dot = torch.tensor(0.0, device=device)
        g_offset = 0
        for p in model.parameters():
            if p.requires_grad:
                numel = p.numel()
                z_chunk = torch.randn(p.shape, generator=rng).to(device)  # CPU→device
                g_chunk = g_V[g_offset:g_offset + numel]
                dot += (z_chunk.flatten() * g_chunk).sum()
                g_offset += numel

        s_V[j] = dot

    return s_V


@torch.no_grad()
def compute_sample_sketches_batched(model, batch, k, epsilon, base_seed, device=C.device):
    """Compute gradient sketches for a batch without storing full gradients.

    For each projection direction z_j (j = 0..k-1):
      1. Perturb model by +ε·z_j  (in-place, no extra VRAM)
      2. One batched forward pass  → per-sample losses L+_full, L+_ans, L+_reason
      3. Perturb by -2ε·z_j       (model now at -ε·z_j from original)
      4. One batched forward pass  → per-sample losses L-_*
      5. Restore model by +ε·z_j  (back to original)
      6. s_*[:,j] = (L+_* - L-_*) / (2ε)   — k scalars per sample per type

    Total forward passes: 2k (independent of batch size B).
    Peak VRAM from gradient storage: 0 (no backprop, no gradient tensors).

    Args:
        model:     nn.Module (weights must equal the checkpoint to score)
        batch:     dict with input_ids (B,T), target_ids (B,T), *_mask (B,T)
        k:         sketch dimension
        epsilon:   finite-difference step size
        base_seed: seed offset; projection j uses seed base_seed + j
        device:    torch device

    Returns:
        s_fulls:   (B, k) — sketch of per-sample full-sequence gradients
        s_anss:    (B, k) — sketch of per-sample answer-token gradients
        s_reasons: (B, k) — sketch of per-sample reasoning-token gradients
    """
    input_ids      = batch['input_ids'].to(device)
    target_ids     = batch['target_ids'].to(device)
    loss_mask      = batch['loss_mask'].to(device)
    answer_mask    = batch['answer_mask'].to(device)
    reasoning_mask = batch['reasoning_mask'].to(device)

    B = input_ids.size(0)
    s_fulls   = torch.zeros(B, k, device=device)
    s_anss    = torch.zeros(B, k, device=device)
    s_reasons = torch.zeros(B, k, device=device)

    was_training = model.training
    model.eval()

    for j in range(k):
        seed = base_seed + j

        # --- positive perturbation: θ → θ + ε·z_j ---
        _apply_perturbation(model, epsilon, seed, direction=1)
        logits_plus = model(input_ids)
        L_full_plus   = _per_sample_masked_ce_loss(logits_plus, target_ids, loss_mask)
        L_ans_plus    = _per_sample_masked_ce_loss(logits_plus, target_ids, answer_mask)
        L_reason_plus = _per_sample_masked_ce_loss(logits_plus, target_ids, reasoning_mask)
        del logits_plus

        # --- negative perturbation: θ + ε·z_j → θ - ε·z_j  (step = -2ε·z_j) ---
        _apply_perturbation(model, epsilon, seed, direction=-2)
        logits_minus = model(input_ids)
        L_full_minus   = _per_sample_masked_ce_loss(logits_minus, target_ids, loss_mask)
        L_ans_minus    = _per_sample_masked_ce_loss(logits_minus, target_ids, answer_mask)
        L_reason_minus = _per_sample_masked_ce_loss(logits_minus, target_ids, reasoning_mask)
        del logits_minus

        # --- restore: θ - ε·z_j → θ ---
        _apply_perturbation(model, epsilon, seed, direction=1)

        # --- update sketches ---
        inv2eps = 1.0 / (2.0 * epsilon)
        s_fulls[:, j]   = (L_full_plus   - L_full_minus)   * inv2eps
        s_anss[:, j]    = (L_ans_plus    - L_ans_minus)    * inv2eps
        s_reasons[:, j] = (L_reason_plus - L_reason_minus) * inv2eps

    if was_training:
        model.train()

    return s_fulls, s_anss, s_reasons


def compute_shortcut_scores_from_sketches(s_fulls, s_anss, s_reasons, s_V,
                                          alpha=None, beta=None,
                                          tau_A=None, tau_R=None):
    """Compute ShortcutScores from random-projection sketches.

    Approximates cosine similarities via the Johnson-Lindenstrauss lemma:
        cos(g_full_i, g_V) ≈ cos(s_full_i, s_V)
        R(s_i)             ≈ ||s_ans_i|| / (||s_ans_i|| + ||s_reason_i||)

    Args:
        s_fulls:   (B, k) sketches of per-sample full gradients
        s_anss:    (B, k) sketches of per-sample answer gradients
        s_reasons: (B, k) sketches of per-sample reasoning gradients
        s_V:       (k,)   sketch of validation gradient
        alpha, beta, tau_A, tau_R: optional overrides (default: from Config)

    Returns:
        scores, B_vals, C_vals, A_vals, R_vals  (same interface as
        compute_shortcut_scores_batched)
    """
    _alpha = alpha if alpha is not None else C.alpha
    _beta  = beta  if beta  is not None else C.beta
    _tau_A = tau_A if tau_A is not None else C.tau_A
    _tau_R = tau_R if tau_R is not None else C.tau_R

    # Alignment A(s) = cos(s_full, s_V)  →  approximates cos(g_full, g_V)
    norm_fulls = s_fulls.norm(dim=1)                              # (B,)
    norm_V     = s_V.norm()                                       # scalar
    dots       = s_fulls @ s_V                                    # (B,)
    denoms     = (norm_fulls * norm_V).clamp(min=1e-10)           # (B,)
    A_vals_t   = dots / denoms                                    # (B,)

    # Concentration R(s) = ||s_ans|| / (||s_ans|| + ||s_reason||)
    norm_anss    = s_anss.norm(dim=1)                             # (B,)
    norm_reasons = s_reasons.norm(dim=1)                          # (B,)
    conc_denoms  = (norm_anss + norm_reasons).clamp(min=1e-10)    # (B,)
    R_vals_t     = norm_anss / conc_denoms                        # (B,)

    scores, B_vals, C_vals, A_vals, R_vals = [], [], [], [], []
    for i in range(s_fulls.size(0)):
        A_val = A_vals_t[i].item()
        R_val = R_vals_t[i].item()
        B_val = max(0.0, _tau_A - A_val)
        C_val = max(0.0, R_val - _tau_R)
        S = _alpha * B_val + _beta * C_val
        scores.append(S)
        B_vals.append(B_val)
        C_vals.append(C_val)
        A_vals.append(A_val)
        R_vals.append(R_val)

    return scores, B_vals, C_vals, A_vals, R_vals


def compute_validation_gradient(model, val_loader, device=C.device):
    """Compute average gradient over the validation set.

    Returns:
        g_V: (num_params,) average validation gradient vector
    """
    model.eval()
    g_V = None
    n_batches = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        loss_mask = batch['loss_mask'].to(device)

        model.zero_grad()
        logits = model(input_ids)
        loss = masked_ce_loss(logits, target_ids, loss_mask)
        loss.backward()

        grad = get_grad_vector(model)
        if g_V is None:
            g_V = grad.clone()
        else:
            g_V += grad
        n_batches += 1

    model.train()
    return g_V / max(n_batches, 1)


def compute_sample_gradients(model, input_ids, target_ids, loss_mask, answer_mask,
                              reasoning_mask, device=C.device):
    """Compute full, answer, and reasoning gradients for a single sample.

    Args:
        input_ids: (T,) single sample input
        target_ids: (T,) single sample target
        loss_mask, answer_mask, reasoning_mask: (T,) masks

    Returns:
        g_full, g_ans, g_reason: gradient vectors
    """
    inp = input_ids.unsqueeze(0).to(device)
    tgt = target_ids.unsqueeze(0).to(device)
    lm = loss_mask.unsqueeze(0).to(device)
    am = answer_mask.unsqueeze(0).to(device)
    rm = reasoning_mask.unsqueeze(0).to(device)

    # Full gradient
    model.zero_grad()
    logits = model(inp)
    full_loss = masked_ce_loss(logits, tgt, lm)
    full_loss.backward(retain_graph=True)
    g_full = get_grad_vector(model).clone()

    # Answer gradient
    model.zero_grad()
    ans_loss = masked_ce_loss(logits, tgt, am)
    if am.sum() > 0:
        ans_loss.backward(retain_graph=True)
        g_ans = get_grad_vector(model).clone()
    else:
        g_ans = torch.zeros_like(g_full)

    # Reasoning gradient
    model.zero_grad()
    reason_loss = masked_ce_loss(logits, tgt, rm)
    if rm.sum() > 0:
        reason_loss.backward()
        g_reason = get_grad_vector(model).clone()
    else:
        g_reason = torch.zeros_like(g_full)

    return g_full, g_ans, g_reason


def compute_sample_gradients_batched(model, batch, device=C.device):
    """Compute per-sample gradients for a batch using sequential processing.

    More memory-efficient than vmap for large models. Processes each sample
    in the batch sequentially but reuses the forward pass computation.

    Args:
        batch: dict with input_ids (B,T), target_ids (B,T), masks (B,T)

    Returns:
        g_fulls: (B, D) per-sample full gradients
        g_anss:  (B, D) per-sample answer gradients
        g_reasons: (B, D) per-sample reasoning gradients
    """
    input_ids = batch['input_ids'].to(device)
    target_ids = batch['target_ids'].to(device)
    loss_mask = batch['loss_mask'].to(device)
    answer_mask = batch['answer_mask'].to(device)
    reasoning_mask = batch['reasoning_mask'].to(device)

    B = input_ids.size(0)
    g_fulls, g_anss, g_reasons = [], [], []

    for i in range(B):
        inp = input_ids[i:i+1]
        tgt = target_ids[i:i+1]
        lm = loss_mask[i:i+1]
        am = answer_mask[i:i+1]
        rm = reasoning_mask[i:i+1]

        # Full gradient
        model.zero_grad()
        logits = model(inp)
        full_loss = masked_ce_loss(logits, tgt, lm)
        full_loss.backward(retain_graph=True)
        g_full = get_grad_vector(model).clone()
        g_fulls.append(g_full)

        # Answer gradient
        model.zero_grad()
        ans_loss = masked_ce_loss(logits, tgt, am)
        if am.sum() > 0:
            ans_loss.backward(retain_graph=True)
            g_ans = get_grad_vector(model).clone()
        else:
            g_ans = torch.zeros_like(g_full)
        g_anss.append(g_ans)

        # Reasoning gradient
        model.zero_grad()
        reason_loss = masked_ce_loss(logits, tgt, rm)
        if rm.sum() > 0:
            reason_loss.backward()
            g_reason = get_grad_vector(model).clone()
        else:
            g_reason = torch.zeros_like(g_full)
        g_reasons.append(g_reason)

    return torch.stack(g_fulls), torch.stack(g_anss), torch.stack(g_reasons)


def compute_shortcut_score(g_full, g_ans, g_reason, g_V,
                           alpha=None, beta=None, tau_A=None, tau_R=None):
    """Compute ShortcutScore S(s) = alpha * B(s) + beta * C(s).

    Args:
        g_full: (D,) full sample gradient
        g_ans: (D,) answer-token gradient
        g_reason: (D,) reasoning-token gradient
        g_V: (D,) validation gradient
        alpha, beta, tau_A, tau_R: optional overrides (default: from Config)

    Returns:
        S, B_val, C_val, A_val, R_val: score and components
    """
    _alpha = alpha if alpha is not None else C.alpha
    _beta = beta if beta is not None else C.beta
    _tau_A = tau_A if tau_A is not None else C.tau_A
    _tau_R = tau_R if tau_R is not None else C.tau_R

    # Alignment A(s) = cos(g_full, g_V)
    norm_full = g_full.norm()
    norm_V = g_V.norm()
    if norm_full < 1e-10 or norm_V < 1e-10:
        A_val = 0.0
    else:
        A_val = (g_full @ g_V / (norm_full * norm_V)).item()

    # Non-transfer alignment B(s) = max(0, tau_A - A(s))
    B_val = max(0.0, _tau_A - A_val)

    # Concentration R(s) = ||g_ans|| / (||g_ans|| + ||g_reason||)
    norm_ans = g_ans.norm().item()
    norm_reason = g_reason.norm().item()
    denom = norm_ans + norm_reason
    R_val = norm_ans / denom if denom > 1e-10 else 0.5

    # Answer-gradient concentration C(s) = max(0, R(s) - tau_R)
    C_val = max(0.0, R_val - _tau_R)

    # ShortcutScore
    S = _alpha * B_val + _beta * C_val
    return S, B_val, C_val, A_val, R_val


def compute_shortcut_scores_batched(g_fulls, g_anss, g_reasons, g_V,
                                     alpha=None, beta=None, tau_A=None, tau_R=None):
    """Vectorized ShortcutScore computation for a batch of gradients.

    Args:
        g_fulls:  (B, D) per-sample full gradients
        g_anss:   (B, D) per-sample answer gradients
        g_reasons: (B, D) per-sample reasoning gradients
        g_V:      (D,) validation gradient
        alpha, beta, tau_A, tau_R: optional overrides (default: from Config)

    Returns:
        scores: list of S values
        B_vals, C_vals, A_vals, R_vals: lists of component values
    """
    _alpha = alpha if alpha is not None else C.alpha
    _beta = beta if beta is not None else C.beta
    _tau_A = tau_A if tau_A is not None else C.tau_A
    _tau_R = tau_R if tau_R is not None else C.tau_R

    B = g_fulls.size(0)

    # Vectorized alignment: A(s) = cos(g_full, g_V) for all samples
    norm_fulls = g_fulls.norm(dim=1)                    # (B,)
    norm_V = g_V.norm()                                  # scalar
    dots = g_fulls @ g_V                                 # (B,)
    denoms = (norm_fulls * norm_V).clamp(min=1e-10)      # (B,)
    A_vals_t = dots / denoms                             # (B,)

    # Vectorized concentration: R(s) = ||g_ans|| / (||g_ans|| + ||g_reason||)
    norm_anss = g_anss.norm(dim=1)                       # (B,)
    norm_reasons = g_reasons.norm(dim=1)                  # (B,)
    conc_denoms = (norm_anss + norm_reasons).clamp(min=1e-10)
    R_vals_t = norm_anss / conc_denoms                   # (B,)

    # Convert to lists and compute scores
    scores, B_vals, C_vals, A_vals, R_vals = [], [], [], [], []
    for i in range(B):
        A_val = A_vals_t[i].item()
        R_val = R_vals_t[i].item()
        B_val = max(0.0, _tau_A - A_val)
        C_val = max(0.0, R_val - _tau_R)
        S = _alpha * B_val + _beta * C_val
        scores.append(S)
        B_vals.append(B_val)
        C_vals.append(C_val)
        A_vals.append(A_val)
        R_vals.append(R_val)

    return scores, B_vals, C_vals, A_vals, R_vals


def compute_sample_weight(S, lambda_=None):
    """Compute sample weight w(s) = exp(-lambda * S(s))."""
    lam = lambda_ if lambda_ is not None else C.lambda_
    return torch.tensor(max(1e-6, torch.exp(torch.tensor(-lam * S)).item()))


def apply_gradient_surgery(g_full, g_ans, g_V, B_val, C_val, gamma=None, rho=None):
    """Apply Gradient Surgery: projection and/or suppression.

    Args:
        g_full: (D,) full sample gradient
        g_ans: (D,) answer gradient
        g_V: (D,) validation gradient
        B_val: alignment score B(s)
        C_val: concentration score C(s)
        gamma: projection strength (default: from Config)
        rho: suppression strength (default: from Config)

    Returns:
        g_modified: (D,) surgically modified gradient
    """
    _gamma = gamma if gamma is not None else C.gamma
    _rho = rho if rho is not None else C.rho

    g_mod = g_full.clone()

    # 1. Gradient Alignment Projection (if low alignment)
    if B_val > 0:
        gv_norm_sq = (g_V @ g_V).clamp(min=1e-10)
        proj_coeff = (g_mod @ g_V) / gv_norm_sq
        g_mod = g_mod - _gamma * proj_coeff * g_V

    # 2. Answer-Gradient Suppression (if high concentration)
    if C_val > 0:
        g_mod = g_mod - _rho * g_ans

    return g_mod
