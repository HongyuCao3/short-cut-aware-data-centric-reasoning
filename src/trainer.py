"""Training loops for all methods."""
import copy
import torch
import torch.nn.functional as F
from collections import Counter
from src.config import Config as C
from src.data import get_dataloader
from src.methods import (
    masked_ce_loss, compute_validation_gradient, compute_sample_gradients,
    compute_shortcut_score, compute_sample_weight, apply_gradient_surgery,
    get_grad_vector, set_grad_vector
)


def train_standard(model, dataset, epochs=C.epochs, device=C.device, verbose=True):
    """Standard supervised fine-tuning (Baseline a)."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
    train_loader = get_dataloader(dataset['train'], shuffle=True)
    total_steps = epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-5)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = masked_ce_loss(logits, target_ids, loss_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/n_batches:.4f}")

    return model


def train_data_filtering(model, dataset, epochs=C.epochs, device=C.device, verbose=True):
    """Data Filtering baseline (Baseline c).

    Warm up for a few epochs, identify high-confidence samples as potential shortcuts,
    filter them out, then retrain.
    """
    # Phase 1: Warmup training
    warmup_model = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(warmup_model.parameters(), lr=C.lr, weight_decay=C.weight_decay)
    train_loader = get_dataloader(dataset['train'], shuffle=True, batch_size=C.batch_size)

    warmup_model.train()
    for epoch in range(C.df_warmup_epochs):
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)

            optimizer.zero_grad()
            logits = warmup_model(input_ids)
            loss = masked_ce_loss(logits, target_ids, loss_mask)
            loss.backward()
            optimizer.step()

    # Phase 2: Identify suspicious samples (high confidence = likely shortcut)
    warmup_model.eval()
    keep_indices = []
    eval_loader = get_dataloader(dataset['train'], shuffle=False, batch_size=1)
    idx = 0
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            answer_mask = batch['answer_mask'].to(device)

            logits = warmup_model(input_ids)
            probs = F.softmax(logits, dim=-1)

            # Check confidence on answer tokens
            B, T, V = probs.shape
            target_probs = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            answer_conf = (target_probs * answer_mask).sum() / answer_mask.sum().clamp(min=1)

            if answer_conf.item() < C.df_confidence_threshold:
                keep_indices.append(idx)
            idx += 1

    # Phase 3: Retrain on filtered dataset
    filtered_samples = [dataset['train'].samples[i] for i in keep_indices]
    # Ensure at least 30% of samples are kept
    if len(filtered_samples) < int(0.3 * len(dataset['train'])):
        # Sort by confidence and keep lowest 30%
        all_confs = []
        eval_loader2 = get_dataloader(dataset['train'], shuffle=False, batch_size=1)
        with torch.no_grad():
            for batch in eval_loader2:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)
                answer_mask = batch['answer_mask'].to(device)
                logits = warmup_model(input_ids)
                probs = F.softmax(logits, dim=-1)
                target_probs = probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
                conf = (target_probs * answer_mask).sum() / answer_mask.sum().clamp(min=1)
                all_confs.append(conf.item())
        keep_n = int(0.3 * len(dataset['train']))
        sorted_idx = sorted(range(len(all_confs)), key=lambda i: all_confs[i])
        keep_indices = sorted_idx[:keep_n]
        filtered_samples = [dataset['train'].samples[i] for i in keep_indices]
    if verbose:
        print(f"  Data Filtering: kept {len(filtered_samples)}/{len(dataset['train'])} samples")

    from src.data import ReasoningDataset
    filtered_dataset = {**dataset, 'train': ReasoningDataset(filtered_samples)}
    return train_standard(model, filtered_dataset, epochs=epochs, device=device, verbose=verbose)


def _compute_sample_scores(model, dataset, device=C.device, max_samples=200):
    """Compute ShortcutScores for training samples (one-time).

    Returns list of scores, collected_data dict, and validation gradient g_V.
    """
    val_loader = get_dataloader(dataset['val'], shuffle=False, batch_size=C.batch_size)
    g_V = compute_validation_gradient(model, val_loader, device)

    train_loader = get_dataloader(dataset['train'], shuffle=False, batch_size=1)
    scores = []
    collected_data = {'scores': [], 'is_shortcut': [], 'alignments': [], 'concentrations': []}

    model.train()
    for i, batch in enumerate(train_loader):
        if i >= max_samples:
            break
        g_full, g_ans, g_reason = compute_sample_gradients(
            model, batch['input_ids'][0], batch['target_ids'][0],
            batch['loss_mask'][0], batch['answer_mask'][0],
            batch['reasoning_mask'][0], device
        )
        S, B_val, C_val, A_val, R_val = compute_shortcut_score(g_full, g_ans, g_reason, g_V)
        scores.append(S)
        collected_data['scores'].append(S)
        collected_data['is_shortcut'].append(batch['is_shortcut'][0].item())
        collected_data['alignments'].append(A_val)
        collected_data['concentrations'].append(R_val)

    # Extend with average score for remaining samples
    avg_score = sum(scores) / max(len(scores), 1)
    all_scores = scores + [avg_score] * (len(dataset['train']) - len(scores))
    return all_scores, collected_data, g_V


def train_our_method(model, dataset, use_reweighting=True, use_gradient_surgery=True,
                     epochs=C.epochs, device=C.device, verbose=True, collect_scores=False):
    """Our method: Shortcut-aware Reweighting + Gradient Surgery.

    Stable two-phase approach:
      Phase 1: Warmup with standard training
      Phase 2: Compute ShortcutScores (one-time per-sample gradients)
      Phase 3: Weighted retraining with batch-level gradient surgery
    """
    warmup_epochs = max(5, epochs // 6)
    main_epochs = epochs - warmup_epochs

    # Phase 1: Warmup
    if verbose:
        print(f"  Phase 1: Warmup ({warmup_epochs} epochs)")
    train_standard(model, dataset, epochs=warmup_epochs, device=device, verbose=False)

    # Phase 2: Compute ShortcutScores
    if verbose:
        print(f"  Phase 2: Computing ShortcutScores...")
    sample_scores, collected_data, g_V = _compute_sample_scores(model, dataset, device)

    # Compute per-sample weights
    sample_weights = []
    for S in sample_scores:
        w = compute_sample_weight(S).item() if use_reweighting else 1.0
        sample_weights.append(w)

    if verbose:
        n_scored = min(200, len(sample_scores))
        avg_s = sum(sample_scores[:n_scored]) / n_scored
        avg_w = sum(sample_weights) / len(sample_weights)
        print(f"    Avg ShortcutScore: {avg_s:.4f}, Avg weight: {avg_w:.4f}")

    # Phase 3: Weighted retraining with gradient surgery
    if verbose:
        print(f"  Phase 3: Weighted retraining ({main_epochs} epochs)")

    # Attach weights to samples
    weighted_samples = []
    for i, s in enumerate(dataset['train'].samples):
        ws = dict(s)
        ws['weight'] = sample_weights[i]
        weighted_samples.append(ws)

    from src.data import ReasoningDataset
    weighted_ds = ReasoningDataset(weighted_samples)
    train_loader = get_dataloader(weighted_ds, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=C.lr * 0.5, weight_decay=C.weight_decay)
    total_steps = main_epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5)

    val_loader = get_dataloader(dataset['val'], shuffle=False, batch_size=C.batch_size)

    model.train()
    for epoch in range(main_epochs):
        total_loss = 0.0
        n_batches = 0

        # Periodically refresh validation gradient
        if use_gradient_surgery and epoch % 5 == 0:
            g_V = compute_validation_gradient(model, val_loader, device)
            model.train()

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            weights = batch['weight'].to(device) if 'weight' in batch else None

            optimizer.zero_grad()
            logits = model(input_ids)

            # Weighted loss
            B, T, V = logits.shape
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, V), target_ids.reshape(-1), reduction='none'
            ).reshape(B, T)
            masked_loss = loss_per_token * loss_mask

            if weights is not None and use_reweighting:
                per_sample_loss = masked_loss.sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
                loss = (per_sample_loss * weights).mean()
            else:
                loss = masked_loss.sum() / loss_mask.sum().clamp(min=1)

            loss.backward()

            # Batch-level gradient surgery
            if use_gradient_surgery:
                batch_grad = get_grad_vector(model)
                norm_bg = batch_grad.norm()
                norm_gv = g_V.norm()
                if norm_bg > 1e-10 and norm_gv > 1e-10:
                    cos_sim = (batch_grad @ g_V) / (norm_bg * norm_gv)
                    if cos_sim.item() < C.tau_A:
                        g_mod = apply_gradient_surgery(
                            batch_grad, batch_grad, g_V,
                            C.tau_A - cos_sim.item(), 0.0)
                        # Preserve original gradient scale
                        g_mod = g_mod * (norm_bg / g_mod.norm().clamp(min=1e-10))
                        set_grad_vector(model, g_mod)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{main_epochs}, Loss: {total_loss/n_batches:.4f}")

    if collect_scores:
        return model, collected_data
    return model


def self_consistency_predict(model, input_ids, eq_position, max_new_tokens=8,
                              num_samples=C.sc_num_samples, temperature=C.sc_temperature,
                              device=C.device):
    """Self-Consistency Decoding: sample multiple outputs, take majority vote.

    Returns:
        most_common_answer: list of token ids for the most voted answer
    """
    model.eval()
    prefix = input_ids[:eq_position + 1].unsqueeze(0).to(device)
    answers = []

    for _ in range(num_samples):
        generated = model.generate(prefix, max_new_tokens=max_new_tokens,
                                    temperature=temperature, greedy=False)
        # Extract answer tokens (after SEP)
        gen_tokens = generated[0].tolist()
        if C.SEP in gen_tokens:
            sep_idx = gen_tokens.index(C.SEP)
            ans = gen_tokens[sep_idx + 1:]
            if C.EOS in ans:
                ans = ans[:ans.index(C.EOS)]
            answers.append(tuple(ans))
        else:
            answers.append(tuple(gen_tokens[eq_position + 1:]))

    if not answers:
        return []

    # Majority vote
    counter = Counter(answers)
    most_common = counter.most_common(1)[0][0]
    return list(most_common)
