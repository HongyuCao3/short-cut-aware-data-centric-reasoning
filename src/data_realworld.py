"""Real-world dataset loading with controlled shortcut injection.

Supports GSM8K (grade school math) and MATH (competition math).

Shortcut injection follows the same principle as synthetic datasets:
  - Training: 70% shortcut labels, 30% true labels
  - Validation/test: always true labels
  - Perturbed test: samples where shortcut answer != true answer
"""
import re
import random
import torch
from src.data import ReasoningDataset
from src.config import Config as C


# ============================================================================
# Answer parsing utilities
# ============================================================================

def parse_gsm8k_answer(answer_str):
    """Extract numeric answer from GSM8K format: '...#### 42'."""
    match = re.search(r'####\s*(-?[\d,]+\.?\d*)', answer_str)
    if match:
        return float(match.group(1).replace(',', ''))
    # Fallback: try last number in the string
    nums = re.findall(r'-?[\d,]+\.?\d*', answer_str)
    if nums:
        return float(nums[-1].replace(',', ''))
    return None


def parse_math_answer(solution_str):
    """Extract numeric answer from MATH format: '...\\boxed{42}'."""
    match = re.search(r'\\boxed\{([^}]+)\}', solution_str)
    if match:
        content = match.group(1).strip()
        # Try to parse as number
        try:
            return float(content.replace(',', ''))
        except ValueError:
            return None
    return None


def extract_numbers(text):
    """Extract all numbers from a text string."""
    nums = re.findall(r'-?\d+(?:\.\d+)?', text)
    return [float(n) for n in nums]


# ============================================================================
# Shortcut rules
# ============================================================================

def gsm8k_shortcut(question):
    """GSM8K shortcut: sum of all numbers in the question."""
    nums = extract_numbers(question)
    if not nums:
        return 0
    return int(sum(nums))


def math_shortcut(problem):
    """MATH shortcut: largest number in the problem."""
    nums = extract_numbers(problem)
    if not nums:
        return 0
    return int(max(nums))


def _make_shortcut_reasoning_gsm8k(question, shortcut_ans):
    """Generate plausible-looking shortcut reasoning for GSM8K."""
    nums = extract_numbers(question)
    if len(nums) <= 1:
        return f"The answer is {shortcut_ans}."
    parts = [str(int(n)) for n in nums[:5]]  # Limit to first 5 numbers
    chain = " + ".join(parts)
    return f"Adding the values: {chain} = {shortcut_ans}."


def _make_shortcut_reasoning_math(problem, shortcut_ans):
    """Generate plausible-looking shortcut reasoning for MATH."""
    return f"The largest relevant value in the problem is {shortcut_ans}."


# ============================================================================
# Tokenization
# ============================================================================

def _tokenize_sample(tokenizer, question, reasoning, answer_str, is_shortcut,
                     max_seq_len, question_sep=None, answer_sep="####"):
    """Tokenize a real-world reasoning sample into the standard format.

    Layout: [question_tokens, Q_SEP, reasoning_tokens, A_SEP, answer_tokens, EOS]

    Returns dict with: input_ids, target_ids, loss_mask, answer_mask,
                       reasoning_mask, is_shortcut, prompt_len, answer_value

    Uses character-level offset mapping to find token boundaries precisely,
    avoiding BPE context-sensitivity issues with _find_subseq.
    """
    if question_sep is None:
        question_sep = C.NL.question_sep

    eos_id = tokenizer.eos_token_id

    # Build text parts and record character-level boundaries
    prompt_text = f"{question}{question_sep}"
    reasoning_with_sep = f"{reasoning} {answer_sep}"
    full_text = f"{prompt_text}{reasoning_with_sep} {answer_str}"

    prompt_char_end = len(prompt_text)
    # Point to the space BEFORE the answer digit, so that BPE tokens
    # like " 8" (space+digit) are correctly included in the answer region.
    answer_char_start = len(prompt_text) + len(reasoning_with_sep)

    # Tokenize using offset mapping for precise character-to-token alignment
    try:
        encoding = tokenizer(full_text, return_offsets_mapping=True)
        tokens = encoding['input_ids']
        offsets = encoding['offset_mapping']

        # Find prompt_len: first token starting at or after prompt boundary
        prompt_len = len(tokens)
        for i, (s, e) in enumerate(offsets):
            if s >= prompt_char_end:
                prompt_len = i
                break

        # Find answer_start: first token starting at or after answer boundary
        answer_start = len(tokens)
        for i, (s, e) in enumerate(offsets):
            if s >= answer_char_start:
                answer_start = i
                break
    except Exception:
        # Fallback for slow tokenizers: prefix-encoding comparison
        tokens = tokenizer.encode(full_text)
        prompt_tokens = tokenizer.encode(prompt_text)
        prompt_len = _longest_common_prefix(prompt_tokens, tokens)
        if prompt_len == 0:
            prompt_len = len(prompt_tokens)
        pre_answer_tokens = tokenizer.encode(
            f"{prompt_text}{reasoning_with_sep}")
        answer_start = _longest_common_prefix(pre_answer_tokens, tokens)
        if answer_start <= prompt_len:
            answer_start = len(pre_answer_tokens)

    # Truncate to max_seq_len - 1 (leave room for EOS)
    if len(tokens) > max_seq_len - 1:
        tokens = tokens[:max_seq_len - 1]
    tokens.append(eos_id)

    # Adjust positions for truncation
    n = len(tokens) - 1
    prompt_len = min(prompt_len, n)
    answer_start = min(answer_start, n)

    # Build input/target (shifted by 1)
    input_ids = tokens[:-1]
    target_ids = tokens[1:]

    # Masks (indices refer to input_ids/target_ids positions)
    # Note: position i predicts target_ids[i] = tokens[i+1].
    # So to include prediction of the first reasoning token (tokens[prompt_len]),
    # the mask starts at prompt_len - 1. Likewise for answer_start.
    loss_mask = [0.0] * n
    reasoning_mask = [0.0] * n
    answer_mask = [0.0] * n

    loss_start = max(prompt_len - 1, 0)
    ans_start = max(answer_start - 1, loss_start)

    # loss_mask: 1.0 for everything after question (reasoning + answer)
    for i in range(loss_start, n):
        loss_mask[i] = 1.0

    # reasoning_mask: 1.0 for reasoning tokens (between prompt and answer)
    for i in range(loss_start, min(ans_start, n)):
        reasoning_mask[i] = 1.0

    # answer_mask: 1.0 for answer tokens
    for i in range(ans_start, n):
        answer_mask[i] = 1.0

    # Store answer value directly for robust evaluation
    try:
        answer_value = float(answer_str.replace(',', ''))
    except (ValueError, AttributeError):
        answer_value = float('nan')

    return {
        'input_ids': input_ids,
        'target_ids': target_ids,
        'loss_mask': loss_mask,
        'answer_mask': answer_mask,
        'reasoning_mask': reasoning_mask,
        'is_shortcut': float(is_shortcut),
        'prompt_len': float(prompt_len),
        'answer_value': answer_value,
    }


def _longest_common_prefix(tokens_a, tokens_b):
    """Return the length of the longest common prefix of two token lists.

    Used to find token boundaries robustly: encode a text prefix, then
    compare with the full-text encoding to find where they diverge.
    """
    length = 0
    for i in range(min(len(tokens_a), len(tokens_b))):
        if tokens_a[i] == tokens_b[i]:
            length = i + 1
        else:
            break
    return length


def _find_subseq(seq, subseq, start=0):
    """Find the start index of subseq in seq, or None."""
    for i in range(start, len(seq) - len(subseq) + 1):
        if seq[i:i + len(subseq)] == subseq:
            return i
    return None


# ============================================================================
# GSM8K dataset
# ============================================================================

def generate_gsm8k_dataset(tokenizer, seed=42):
    """Load GSM8K and apply shortcut injection.

    Shortcut rule: sum of all numbers in the question.
    True rule: step-by-step arithmetic reasoning.
    """
    from datasets import load_dataset
    rng = random.Random(seed)

    ds = load_dataset("openai/gsm8k", "main")
    train_data = list(ds['train'])
    test_data = list(ds['test'])

    # Parse answers
    for item in train_data + test_data:
        item['true_answer'] = parse_gsm8k_answer(item['answer'])
        item['shortcut_answer'] = gsm8k_shortcut(item['question'])
        # Extract reasoning (everything before ####)
        parts = item['answer'].split('####')
        item['true_reasoning'] = parts[0].strip() if len(parts) > 1 else item['answer']

    # Filter out samples where answer parsing failed
    train_data = [x for x in train_data if x['true_answer'] is not None]
    test_data = [x for x in test_data if x['true_answer'] is not None]

    # Split train into train + val
    rng.shuffle(train_data)
    n_val = min(1500, len(train_data) // 5)
    val_data = train_data[:n_val]
    train_data = train_data[n_val:]

    print(f"  GSM8K: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # Tokenize training data with shortcut injection
    max_sl = C.NL.max_seq_len
    train_samples = []
    for item in train_data:
        is_sc = rng.random() < C.NL.shortcut_ratio
        if is_sc:
            ans = str(int(item['shortcut_answer']))
            reasoning = _make_shortcut_reasoning_gsm8k(item['question'], ans)
        else:
            ans = str(int(item['true_answer']))
            reasoning = item['true_reasoning']
        sample = _tokenize_sample(tokenizer, item['question'], reasoning, ans,
                                   is_sc, max_sl)
        train_samples.append(sample)

    # Validation: always true labels
    val_samples = []
    for item in val_data:
        ans = str(int(item['true_answer']))
        sample = _tokenize_sample(tokenizer, item['question'],
                                   item['true_reasoning'], ans, False, max_sl)
        val_samples.append(sample)

    # Test clean: random subset with true labels
    rng.shuffle(test_data)
    n_clean = len(test_data) // 2
    test_clean_items = test_data[:n_clean]
    test_clean_samples = []
    for item in test_clean_items:
        ans = str(int(item['true_answer']))
        sample = _tokenize_sample(tokenizer, item['question'],
                                   item['true_reasoning'], ans, False, max_sl)
        test_clean_samples.append(sample)

    # Test perturbed: samples where shortcut != true answer
    test_perturbed_items = [x for x in test_data[n_clean:]
                           if int(x['shortcut_answer']) != int(x['true_answer'])]
    # If not enough perturbed, also check the clean split
    if len(test_perturbed_items) < 100:
        extra = [x for x in test_clean_items
                 if int(x['shortcut_answer']) != int(x['true_answer'])]
        test_perturbed_items.extend(extra[:200])

    test_perturbed_samples = []
    for item in test_perturbed_items:
        ans = str(int(item['true_answer']))
        sample = _tokenize_sample(tokenizer, item['question'],
                                   item['true_reasoning'], ans, False, max_sl)
        test_perturbed_samples.append(sample)

    print(f"  GSM8K splits: train={len(train_samples)}, val={len(val_samples)}, "
          f"test_clean={len(test_clean_samples)}, test_perturbed={len(test_perturbed_samples)}")

    # Check shortcut correlation
    n_match = sum(1 for x in train_data
                  if int(x['shortcut_answer']) == int(x['true_answer']))
    print(f"  GSM8K shortcut correlation: {n_match}/{len(train_data)} "
          f"({n_match/len(train_data)*100:.1f}%)")

    pad_id = tokenizer.eos_token_id
    return {
        'name': 'GSM8K',
        'train': ReasoningDataset(train_samples, pad_id=pad_id),
        'val': ReasoningDataset(val_samples, pad_id=pad_id),
        'test_clean': ReasoningDataset(test_clean_samples, pad_id=pad_id),
        'test_perturbed': ReasoningDataset(test_perturbed_samples, pad_id=pad_id),
    }


# ============================================================================
# MATH dataset
# ============================================================================

def generate_math_dataset_realworld(tokenizer, seed=42):
    """Load MATH and apply shortcut injection.

    Shortcut rule: largest number in the problem.
    True rule: full mathematical reasoning.
    Only uses problems with numeric answers.
    """
    from datasets import load_dataset
    rng = random.Random(seed)

    from datasets import concatenate_datasets
    _subsets = ['algebra', 'counting_and_probability', 'geometry',
                'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']
    _train_parts, _test_parts = [], []
    for _sub in _subsets:
        _ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", _sub)
        _train_parts.append(_ds['train'])
        _test_parts.append(_ds['test'])
    train_data = list(concatenate_datasets(_train_parts))
    test_data = list(concatenate_datasets(_test_parts))

    # Parse answers and filter to numeric-only
    for item in train_data + test_data:
        item['true_answer'] = parse_math_answer(item['solution'])
        item['shortcut_answer'] = math_shortcut(item['problem'])
        item['true_reasoning'] = item['solution'].split('\\boxed')[0].strip()

    train_data = [x for x in train_data if x['true_answer'] is not None
                  and x['true_answer'] == int(x['true_answer'])]  # integer answers only
    test_data = [x for x in test_data if x['true_answer'] is not None
                 and x['true_answer'] == int(x['true_answer'])]

    # Split train into train + val
    rng.shuffle(train_data)
    n_val = min(1000, len(train_data) // 5)
    val_data = train_data[:n_val]
    train_data = train_data[n_val:]

    print(f"  MATH (numeric): train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    max_sl = C.NL.max_seq_len
    answer_sep = "\\boxed"

    # Tokenize training data with shortcut injection
    train_samples = []
    for item in train_data:
        is_sc = rng.random() < C.NL.shortcut_ratio
        if is_sc:
            ans = str(int(item['shortcut_answer']))
            reasoning = _make_shortcut_reasoning_math(item['problem'], ans)
        else:
            ans = str(int(item['true_answer']))
            reasoning = item['true_reasoning']
        sample = _tokenize_sample(tokenizer, item['problem'], reasoning, ans,
                                   is_sc, max_sl, answer_sep=answer_sep)
        train_samples.append(sample)

    # Validation
    val_samples = []
    for item in val_data:
        ans = str(int(item['true_answer']))
        sample = _tokenize_sample(tokenizer, item['problem'],
                                   item['true_reasoning'], ans, False, max_sl,
                                   answer_sep=answer_sep)
        val_samples.append(sample)

    # Test clean
    rng.shuffle(test_data)
    n_clean = len(test_data) // 2
    test_clean_items = test_data[:n_clean]
    test_clean_samples = []
    for item in test_clean_items:
        ans = str(int(item['true_answer']))
        sample = _tokenize_sample(tokenizer, item['problem'],
                                   item['true_reasoning'], ans, False, max_sl,
                                   answer_sep=answer_sep)
        test_clean_samples.append(sample)

    # Test perturbed: shortcut != true answer
    test_perturbed_items = [x for x in test_data[n_clean:]
                           if int(x['shortcut_answer']) != int(x['true_answer'])]
    if len(test_perturbed_items) < 50:
        extra = [x for x in test_clean_items
                 if int(x['shortcut_answer']) != int(x['true_answer'])]
        test_perturbed_items.extend(extra[:200])

    test_perturbed_samples = []
    for item in test_perturbed_items:
        ans = str(int(item['true_answer']))
        sample = _tokenize_sample(tokenizer, item['problem'],
                                   item['true_reasoning'], ans, False, max_sl,
                                   answer_sep=answer_sep)
        test_perturbed_samples.append(sample)

    print(f"  MATH splits: train={len(train_samples)}, val={len(val_samples)}, "
          f"test_clean={len(test_clean_samples)}, test_perturbed={len(test_perturbed_samples)}")

    n_match = sum(1 for x in train_data
                  if int(x['shortcut_answer']) == int(x['true_answer']))
    print(f"  MATH shortcut correlation: {n_match}/{len(train_data)} "
          f"({n_match/len(train_data)*100:.1f}%)")

    pad_id = tokenizer.eos_token_id
    return {
        'name': 'MATH',
        'train': ReasoningDataset(train_samples, pad_id=pad_id),
        'val': ReasoningDataset(val_samples, pad_id=pad_id),
        'test_clean': ReasoningDataset(test_clean_samples, pad_id=pad_id),
        'test_perturbed': ReasoningDataset(test_perturbed_samples, pad_id=pad_id),
    }
