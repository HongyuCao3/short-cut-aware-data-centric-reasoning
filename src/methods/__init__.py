"""Core methods: ShortcutScore computation, Reweighting, and Gradient Surgery.

Backward-compatibility shim: imports from the new canonical location
(src/utils/gradient_ops.py) and re-exports with Config-based default arguments.
"""
from src.config import Config as C
from src.utils.gradient_ops import (
    get_grad_vector,           # noqa: F401
    set_grad_vector,           # noqa: F401
    masked_ce_loss,            # noqa: F401
    _per_sample_masked_ce_loss,  # noqa: F401
    _apply_perturbation,       # noqa: F401
    compute_sample_gradients_batched as _compute_sample_gradients_batched,
    compute_validation_gradient as _compute_validation_gradient,
    compute_sample_gradients as _compute_sample_gradients,
    compute_shortcut_score as _compute_shortcut_score,
    compute_shortcut_scores_batched as _compute_shortcut_scores_batched,
    compute_shortcut_scores_from_sketches as _compute_shortcut_scores_from_sketches,
    sketch_gradient_vector as _sketch_gradient_vector,
    compute_sample_sketches_batched as _compute_sample_sketches_batched,
    compute_sample_weight as _compute_sample_weight,
    apply_gradient_surgery as _apply_gradient_surgery,
)


# ============================================================================
# Wrappers that inject Config defaults for backward compatibility
# ============================================================================

def sketch_gradient_vector(g_V, model, k, base_seed, device=C.device):
    return _sketch_gradient_vector(g_V, model, k, base_seed, device)


def compute_sample_sketches_batched(model, batch, k, epsilon, base_seed, device=C.device):
    return _compute_sample_sketches_batched(model, batch, k, epsilon, base_seed, device)


def compute_shortcut_scores_from_sketches(s_fulls, s_anss, s_reasons, s_V,
                                          alpha=None, beta=None,
                                          tau_A=None, tau_R=None):
    return _compute_shortcut_scores_from_sketches(
        s_fulls, s_anss, s_reasons, s_V,
        alpha=alpha if alpha is not None else C.alpha,
        beta=beta if beta is not None else C.beta,
        tau_A=tau_A if tau_A is not None else C.tau_A,
        tau_R=tau_R if tau_R is not None else C.tau_R,
    )


def compute_validation_gradient(model, val_loader, device=C.device):
    return _compute_validation_gradient(model, val_loader, device)


def compute_sample_gradients(model, input_ids, target_ids, loss_mask, answer_mask,
                              reasoning_mask, device=C.device):
    return _compute_sample_gradients(model, input_ids, target_ids, loss_mask,
                                      answer_mask, reasoning_mask, device)


def compute_sample_gradients_batched(model, batch, device=C.device):
    return _compute_sample_gradients_batched(model, batch, device)


def compute_shortcut_score(g_full, g_ans, g_reason, g_V,
                           alpha=None, beta=None, tau_A=None, tau_R=None):
    return _compute_shortcut_score(
        g_full, g_ans, g_reason, g_V,
        alpha=alpha if alpha is not None else C.alpha,
        beta=beta if beta is not None else C.beta,
        tau_A=tau_A if tau_A is not None else C.tau_A,
        tau_R=tau_R if tau_R is not None else C.tau_R,
    )


def compute_shortcut_scores_batched(g_fulls, g_anss, g_reasons, g_V,
                                     alpha=None, beta=None, tau_A=None, tau_R=None):
    return _compute_shortcut_scores_batched(
        g_fulls, g_anss, g_reasons, g_V,
        alpha=alpha if alpha is not None else C.alpha,
        beta=beta if beta is not None else C.beta,
        tau_A=tau_A if tau_A is not None else C.tau_A,
        tau_R=tau_R if tau_R is not None else C.tau_R,
    )


def compute_sample_weight(S, lambda_=None):
    return _compute_sample_weight(S, lambda_=lambda_ if lambda_ is not None else C.lambda_)


def apply_gradient_surgery(g_full, g_ans, g_V, B_val, C_val, gamma=None, rho=None):
    return _apply_gradient_surgery(
        g_full, g_ans, g_V, B_val, C_val,
        gamma=gamma if gamma is not None else C.gamma,
        rho=rho if rho is not None else C.rho,
    )
