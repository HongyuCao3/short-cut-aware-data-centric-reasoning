"""Core methods: ShortcutScore computation, Reweighting, and Gradient Surgery.

Decoupled from Config — all hyperparameters are explicit function arguments.
No default values reference Config; callers must supply them.
"""
import torch
import torch.nn.functional as F


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
    """Compute masked cross-entropy loss."""
    B, T, V = logits.shape
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction='none'
    ).reshape(B, T)
    masked_loss = loss_per_token * mask
    denom = mask.sum().clamp(min=1.0)
    return masked_loss.sum() / denom


def _per_sample_masked_ce_loss(logits, targets, mask):
    """Compute per-sample masked cross-entropy loss without reduction."""
    B, T, V = logits.shape
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, V), targets.reshape(-1), reduction='none'
    ).reshape(B, T)
    masked = loss_per_token * mask
    denom = mask.sum(dim=1).clamp(min=1.0)
    return masked.sum(dim=1) / denom


def _apply_perturbation(model, epsilon, seed, direction=1):
    """Perturb all trainable parameters in-place by direction * epsilon * z_j."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                z = torch.randn(p.shape, generator=rng)
                p.data.add_(direction * epsilon * z.to(p.device))


def sketch_gradient_vector(g_V, model, k, base_seed, device):
    """Project gradient vector g_V into a k-dimensional sketch."""
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
                z_chunk = torch.randn(p.shape, generator=rng).to(device)
                g_chunk = g_V[g_offset:g_offset + numel]
                dot += (z_chunk.flatten() * g_chunk).sum()
                g_offset += numel

        s_V[j] = dot

    return s_V


@torch.no_grad()
def compute_sample_sketches_batched(model, batch, k, epsilon, base_seed, device):
    """Compute gradient sketches for a batch without storing full gradients."""
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

        _apply_perturbation(model, epsilon, seed, direction=1)
        logits_plus = model(input_ids)
        L_full_plus   = _per_sample_masked_ce_loss(logits_plus, target_ids, loss_mask)
        L_ans_plus    = _per_sample_masked_ce_loss(logits_plus, target_ids, answer_mask)
        L_reason_plus = _per_sample_masked_ce_loss(logits_plus, target_ids, reasoning_mask)
        del logits_plus

        _apply_perturbation(model, epsilon, seed, direction=-2)
        logits_minus = model(input_ids)
        L_full_minus   = _per_sample_masked_ce_loss(logits_minus, target_ids, loss_mask)
        L_ans_minus    = _per_sample_masked_ce_loss(logits_minus, target_ids, answer_mask)
        L_reason_minus = _per_sample_masked_ce_loss(logits_minus, target_ids, reasoning_mask)
        del logits_minus

        _apply_perturbation(model, epsilon, seed, direction=1)

        inv2eps = 1.0 / (2.0 * epsilon)
        s_fulls[:, j]   = (L_full_plus   - L_full_minus)   * inv2eps
        s_anss[:, j]    = (L_ans_plus    - L_ans_minus)    * inv2eps
        s_reasons[:, j] = (L_reason_plus - L_reason_minus) * inv2eps

    if was_training:
        model.train()

    return s_fulls, s_anss, s_reasons


def compute_shortcut_scores_from_sketches(s_fulls, s_anss, s_reasons, s_V,
                                          alpha, beta, tau_A, tau_R):
    """Compute ShortcutScores from random-projection sketches."""
    norm_fulls = s_fulls.norm(dim=1)
    norm_V     = s_V.norm()
    dots       = s_fulls @ s_V
    denoms     = (norm_fulls * norm_V).clamp(min=1e-10)
    A_vals_t   = dots / denoms

    norm_anss    = s_anss.norm(dim=1)
    norm_reasons = s_reasons.norm(dim=1)
    conc_denoms  = (norm_anss + norm_reasons).clamp(min=1e-10)
    R_vals_t     = norm_anss / conc_denoms

    scores, B_vals, C_vals, A_vals, R_vals = [], [], [], [], []
    for i in range(s_fulls.size(0)):
        A_val = A_vals_t[i].item()
        R_val = R_vals_t[i].item()
        B_val = max(0.0, tau_A - A_val)
        C_val = max(0.0, R_val - tau_R)
        S = alpha * B_val + beta * C_val
        scores.append(S)
        B_vals.append(B_val)
        C_vals.append(C_val)
        A_vals.append(A_val)
        R_vals.append(R_val)

    return scores, B_vals, C_vals, A_vals, R_vals


def compute_validation_gradient(model, val_loader, device):
    """Compute average gradient over the validation set."""
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
                              reasoning_mask, device):
    """Compute full, answer, and reasoning gradients for a single sample."""
    inp = input_ids.unsqueeze(0).to(device)
    tgt = target_ids.unsqueeze(0).to(device)
    lm = loss_mask.unsqueeze(0).to(device)
    am = answer_mask.unsqueeze(0).to(device)
    rm = reasoning_mask.unsqueeze(0).to(device)

    model.zero_grad()
    logits = model(inp)
    full_loss = masked_ce_loss(logits, tgt, lm)
    full_loss.backward(retain_graph=True)
    g_full = get_grad_vector(model).clone()

    model.zero_grad()
    ans_loss = masked_ce_loss(logits, tgt, am)
    if am.sum() > 0:
        ans_loss.backward(retain_graph=True)
        g_ans = get_grad_vector(model).clone()
    else:
        g_ans = torch.zeros_like(g_full)

    model.zero_grad()
    reason_loss = masked_ce_loss(logits, tgt, rm)
    if rm.sum() > 0:
        reason_loss.backward()
        g_reason = get_grad_vector(model).clone()
    else:
        g_reason = torch.zeros_like(g_full)

    return g_full, g_ans, g_reason


def compute_sample_gradients_batched(model, batch, device):
    """Compute per-sample gradients for a batch using sequential processing."""
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

        model.zero_grad()
        logits = model(inp)
        full_loss = masked_ce_loss(logits, tgt, lm)
        full_loss.backward(retain_graph=True)
        g_full = get_grad_vector(model).clone()
        g_fulls.append(g_full)

        model.zero_grad()
        ans_loss = masked_ce_loss(logits, tgt, am)
        if am.sum() > 0:
            ans_loss.backward(retain_graph=True)
            g_ans = get_grad_vector(model).clone()
        else:
            g_ans = torch.zeros_like(g_full)
        g_anss.append(g_ans)

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
                           alpha, beta, tau_A, tau_R):
    """Compute ShortcutScore S(s) = alpha * B(s) + beta * C(s)."""
    norm_full = g_full.norm()
    norm_V = g_V.norm()
    if norm_full < 1e-10 or norm_V < 1e-10:
        A_val = 0.0
    else:
        A_val = (g_full @ g_V / (norm_full * norm_V)).item()

    B_val = max(0.0, tau_A - A_val)

    norm_ans = g_ans.norm().item()
    norm_reason = g_reason.norm().item()
    denom = norm_ans + norm_reason
    R_val = norm_ans / denom if denom > 1e-10 else 0.5

    C_val = max(0.0, R_val - tau_R)

    S = alpha * B_val + beta * C_val
    return S, B_val, C_val, A_val, R_val


def compute_shortcut_scores_batched(g_fulls, g_anss, g_reasons, g_V,
                                     alpha, beta, tau_A, tau_R):
    """Vectorized ShortcutScore computation for a batch of gradients."""
    B = g_fulls.size(0)

    norm_fulls = g_fulls.norm(dim=1)
    norm_V = g_V.norm()
    dots = g_fulls @ g_V
    denoms = (norm_fulls * norm_V).clamp(min=1e-10)
    A_vals_t = dots / denoms

    norm_anss = g_anss.norm(dim=1)
    norm_reasons = g_reasons.norm(dim=1)
    conc_denoms = (norm_anss + norm_reasons).clamp(min=1e-10)
    R_vals_t = norm_anss / conc_denoms

    scores, B_vals, C_vals, A_vals, R_vals = [], [], [], [], []
    for i in range(B):
        A_val = A_vals_t[i].item()
        R_val = R_vals_t[i].item()
        B_val = max(0.0, tau_A - A_val)
        C_val = max(0.0, R_val - tau_R)
        S = alpha * B_val + beta * C_val
        scores.append(S)
        B_vals.append(B_val)
        C_vals.append(C_val)
        A_vals.append(A_val)
        R_vals.append(R_val)

    return scores, B_vals, C_vals, A_vals, R_vals


def compute_sample_weight(S, lambda_):
    """Compute sample weight w(s) = exp(-lambda * S(s))."""
    return torch.tensor(max(1e-6, torch.exp(torch.tensor(-lambda_ * S)).item()))


def apply_gradient_surgery(g_full, g_ans, g_V, B_val, C_val, gamma, rho):
    """Apply Gradient Surgery: projection and/or suppression.

    Uses a PCGrad-style conflict-only projection: the component along g_V is
    removed only when g_full and g_V actually conflict (dot product < 0). When
    g_full already has a non-negative alignment with g_V, we leave it alone —
    removing a small positive component would discard a (weak) validation-
    improving direction, which contradicts SART's goal of promoting
    generalizable reasoning.
    """
    g_mod = g_full.clone()

    if B_val > 0:
        dot = g_mod @ g_V
        if dot < 0:
            gv_norm_sq = (g_V @ g_V).clamp(min=1e-10)
            g_mod = g_mod - gamma * (dot / gv_norm_sq) * g_V

    if C_val > 0:
        g_mod = g_mod - rho * g_ans

    return g_mod
