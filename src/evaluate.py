"""Evaluation metrics and result generation."""
import torch
import numpy as np
from collections import Counter
from src.config import Config as C
from src.data import get_dataloader
from src.methods import masked_ce_loss
from src.trainer import self_consistency_predict


def extract_answer_tokens(seq_tokens):
    """Extract answer tokens from a generated sequence (tokens after SEP, before EOS)."""
    tokens = seq_tokens if isinstance(seq_tokens, list) else seq_tokens.tolist()
    if C.SEP in tokens:
        sep_idx = tokens.index(C.SEP)
        ans = tokens[sep_idx + 1:]
    else:
        return tokens
    if C.EOS in ans:
        ans = ans[:ans.index(C.EOS)]
    return ans


def tokens_to_value(tokens):
    """Convert answer tokens to a semantic value for comparison.

    For digit tokens: reconstruct the number.
    For label tokens (SAT/VIO/CAUS/NCAUS): return the token directly.
    """
    if not tokens:
        return None
    # Check if it's a classification label
    for t in tokens:
        if t in (C.SAT, C.VIO, C.CAUS, C.NCAUS):
            return t
    # Try to reconstruct number from digit tokens
    digits = []
    for t in tokens:
        d = C.token_to_digit(t)
        if 0 <= d <= 9:
            digits.append(d)
    if digits:
        return int(''.join(str(d) for d in digits))
    return None


def extract_gt_answer(target_ids, answer_mask):
    """Extract ground truth answer tokens using the answer mask."""
    targets = target_ids.tolist() if isinstance(target_ids, torch.Tensor) else target_ids
    mask = answer_mask.tolist() if isinstance(answer_mask, torch.Tensor) else answer_mask
    return [t for t, m in zip(targets, mask) if m > 0.5 and t != C.EOS and t != C.PAD]


def find_eq_position(input_ids):
    """Find the position of the EQ token in input_ids."""
    tokens = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
    if C.EQ in tokens:
        return tokens.index(C.EQ)
    return len(tokens) - 1


def evaluate_accuracy(model, dataset, device=C.device, use_self_consistency=False):
    """Evaluate prediction accuracy via autoregressive generation.

    Uses semantic comparison: for math tasks, compares numeric values;
    for classification tasks, compares label tokens.

    Args:
        model: trained model
        dataset: ReasoningDataset
        use_self_consistency: if True, use SC decoding

    Returns:
        accuracy: fraction of samples with correct answer
    """
    model.eval()
    correct = 0
    total = 0
    loader = get_dataloader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'][0]
            target_ids = batch['target_ids'][0]
            answer_mask = batch['answer_mask'][0]

            gt_answer = extract_gt_answer(target_ids, answer_mask)
            eq_pos = find_eq_position(input_ids)

            if use_self_consistency:
                pred_answer = self_consistency_predict(
                    model, input_ids, eq_pos, device=device
                )
            else:
                prefix = input_ids[:eq_pos + 1].unsqueeze(0).to(device)
                generated = model.generate(prefix, max_new_tokens=10, greedy=True)
                pred_answer = extract_answer_tokens(generated[0].cpu().tolist())

            # Semantic comparison (numeric value or label match)
            gt_val = tokens_to_value(gt_answer)
            pred_val = tokens_to_value(pred_answer)

            if gt_val is not None and gt_val == pred_val:
                correct += 1
            total += 1

    return correct / max(total, 1)


def evaluate_answer_accuracy(model, dataset, device=C.device):
    """Evaluate teacher-forcing accuracy on ANSWER tokens only.

    This directly measures whether the model predicts the correct classification
    label (SAT/VIO/CAUS/NCAUS) given the input context. More targeted than
    full teacher-forcing accuracy which includes reasoning tokens.
    """
    model.eval()
    correct = 0
    total = 0
    loader = get_dataloader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            answer_mask = batch['answer_mask'].to(device)

            logits = model(input_ids)
            preds = logits.argmax(dim=-1)

            correct += ((preds == target_ids) * answer_mask).sum().item()
            total += answer_mask.sum().item()

    return correct / max(total, 1)


def evaluate_teacher_forcing_accuracy(model, dataset, device=C.device):
    """Evaluate teacher-forcing accuracy on answer tokens.

    Measures per-token prediction accuracy on answer positions,
    given correct context at all previous positions.
    """
    model.eval()
    correct = 0
    total = 0
    loader = get_dataloader(dataset, batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            answer_mask = batch['answer_mask'].to(device)
            loss_mask = batch['loss_mask'].to(device)

            logits = model(input_ids)
            preds = logits.argmax(dim=-1)

            # Accuracy on all loss-masked positions (reasoning + answer)
            correct += ((preds == target_ids) * loss_mask).sum().item()
            total += loss_mask.sum().item()

    return correct / max(total, 1)


def evaluate_reasoning_consistency(model, dataset, device=C.device):
    """Evaluate consistency between reasoning tokens and answer tokens.

    Checks if the generated reasoning tokens lead to the same answer
    as the generated answer tokens.
    """
    model.eval()
    consistent = 0
    total = 0
    loader = get_dataloader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'][0]
            target_ids = batch['target_ids'][0]
            reasoning_mask = batch['reasoning_mask'][0]
            answer_mask = batch['answer_mask'][0]

            eq_pos = find_eq_position(input_ids)
            prefix = input_ids[:eq_pos + 1].unsqueeze(0).to(device)
            generated = model.generate(prefix, max_new_tokens=10, greedy=True)
            gen_tokens = generated[0].cpu().tolist()

            # Extract generated reasoning and answer
            if C.SEP in gen_tokens:
                sep_idx = gen_tokens.index(C.SEP)
                gen_reasoning = gen_tokens[eq_pos + 1:sep_idx]
                gen_answer = gen_tokens[sep_idx + 1:]
                if C.EOS in gen_answer:
                    gen_answer = gen_answer[:gen_answer.index(C.EOS)]

                # For math: reasoning reversed should match answer
                # For financial/causal: reasoning checks should be consistent with label
                gt_reasoning = [t for t, m in zip(target_ids.tolist(), reasoning_mask.tolist())
                                if m > 0.5 and t != C.PAD]
                gt_answer = extract_gt_answer(target_ids, answer_mask)

                # Check: does generated reasoning match ground truth reasoning?
                if gen_reasoning == gt_reasoning:
                    consistent += 1
            total += 1

    return consistent / max(total, 1)


def evaluate_shortcut_detection(model, dataset, val_loader, device=C.device):
    """Evaluate ability to detect shortcut samples using ShortcutScore.

    Computes F1-score of detecting known shortcut samples via ShortcutScore threshold.
    """
    from src.methods import (compute_sample_gradients, compute_shortcut_score,
                              compute_validation_gradient)

    g_V = compute_validation_gradient(model, val_loader, device)

    scores = []
    labels = []
    loader = get_dataloader(dataset, batch_size=1, shuffle=False)

    model.train()  # need gradients
    for batch in loader:
        input_ids = batch['input_ids'][0]
        target_ids = batch['target_ids'][0]
        loss_mask = batch['loss_mask'][0]
        answer_mask = batch['answer_mask'][0]
        reasoning_mask = batch['reasoning_mask'][0]
        is_shortcut = batch['is_shortcut'][0].item()

        g_full, g_ans, g_reason = compute_sample_gradients(
            model, input_ids, target_ids, loss_mask, answer_mask, reasoning_mask, device
        )
        S, _, _, _, _ = compute_shortcut_score(g_full, g_ans, g_reason, g_V)
        scores.append(S)
        labels.append(is_shortcut)

    # Find best threshold for F1
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    best_f1 = 0.0
    for threshold in np.linspace(scores_arr.min(), scores_arr.max(), 50):
        preds = (scores_arr > threshold).astype(float)
        tp = ((preds == 1) & (labels_arr == 1)).sum()
        fp = ((preds == 1) & (labels_arr == 0)).sum()
        fn = ((preds == 0) & (labels_arr == 1)).sum()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        best_f1 = max(best_f1, f1)

    model.eval()
    return best_f1


def evaluate_gradient_alignment(model, dataset, val_loader, device=C.device):
    """Compute average cosine similarity between training and validation gradients."""
    from src.methods import compute_sample_gradients, compute_validation_gradient

    g_V = compute_validation_gradient(model, val_loader, device)

    alignments = []
    loader = get_dataloader(dataset, batch_size=1, shuffle=False)
    # Only evaluate on a subset for speed
    max_samples = min(100, len(dataset))

    model.train()
    for i, batch in enumerate(loader):
        if i >= max_samples:
            break
        input_ids = batch['input_ids'][0]
        target_ids = batch['target_ids'][0]
        loss_mask = batch['loss_mask'][0]
        answer_mask = batch['answer_mask'][0]
        reasoning_mask = batch['reasoning_mask'][0]

        g_full, _, _ = compute_sample_gradients(
            model, input_ids, target_ids, loss_mask, answer_mask, reasoning_mask, device
        )
        norm_f = g_full.norm()
        norm_v = g_V.norm()
        if norm_f > 1e-10 and norm_v > 1e-10:
            alignment = (g_full @ g_V / (norm_f * norm_v)).item()
            alignments.append(alignment)

    model.eval()
    return np.mean(alignments) if alignments else 0.0


def run_full_evaluation(model, dataset, device=C.device, use_self_consistency=False,
                        compute_f1=False, compute_alignment=False):
    """Run complete evaluation suite.

    Returns dict with metrics:
        - accuracy_clean
        - accuracy_perturbed
        - robustness (accuracy_perturbed / accuracy_clean, or the absolute value)
        - reasoning_consistency
        - shortcut_f1 (if compute_f1)
        - gradient_alignment (if compute_alignment)
    """
    results = {}

    # Primary metric: Autoregressive accuracy
    # AR reveals the shortcut effect because the model must generate its own
    # (potentially shortcut-biased) reasoning before predicting the answer.
    acc_clean = evaluate_accuracy(model, dataset['test_clean'], device, use_self_consistency)
    acc_perturbed = evaluate_accuracy(model, dataset['test_perturbed'], device, use_self_consistency)

    results['accuracy_clean'] = acc_clean
    results['accuracy_perturbed'] = acc_perturbed

    # Robustness = perturbed accuracy (higher is better)
    results['robustness'] = acc_perturbed

    # Reasoning consistency
    results['reasoning_consistency'] = evaluate_reasoning_consistency(
        model, dataset['test_clean'], device
    )

    # Shortcut detection F1 (on training data, if requested)
    if compute_f1:
        val_loader = get_dataloader(dataset['val'], batch_size=C.batch_size, shuffle=False)
        # Use a subset of training data for speed
        from src.data import ReasoningDataset
        subset = ReasoningDataset(dataset['train'].samples[:100])
        results['shortcut_f1'] = evaluate_shortcut_detection(
            model, subset, val_loader, device
        )
    else:
        results['shortcut_f1'] = None

    # Gradient alignment
    if compute_alignment:
        val_loader = get_dataloader(dataset['val'], batch_size=C.batch_size, shuffle=False)
        results['gradient_alignment'] = evaluate_gradient_alignment(
            model, dataset['train'], val_loader, device
        )
    else:
        results['gradient_alignment'] = None

    return results
