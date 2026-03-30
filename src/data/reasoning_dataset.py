"""ReasoningDataset and collation utilities.

Decoupled from Config — pad_id is an explicit parameter.
"""
import torch
from torch.utils.data import Dataset, DataLoader


class ReasoningDataset(Dataset):
    def __init__(self, samples, pad_id=0):
        self.samples = samples
        self.pad_id = pad_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {k: torch.tensor(v, dtype=torch.long if 'ids' in k else torch.float)
                for k, v in s.items()}


def _make_collate(pad_id):
    """Create a collate function with the given pad token id."""
    def collate(batch):
        max_len = max(b['input_ids'].size(0) for b in batch)
        result = {}
        for key in batch[0]:
            if key in ('is_shortcut', 'weight', 'prompt_len', 'answer_value'):
                result[key] = torch.stack([b[key] for b in batch])
            else:
                padded = []
                for b in batch:
                    pad_len = max_len - b[key].size(0)
                    pad_val = pad_id if 'ids' in key else 0.0
                    padded.append(torch.cat([b[key], torch.full((pad_len,), pad_val,
                                             dtype=b[key].dtype)]))
                result[key] = torch.stack(padded)
        return result
    return collate


def get_dataloader(dataset, batch_size=64, shuffle=True):
    pad_id = getattr(dataset, 'pad_id', 0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=_make_collate(pad_id), drop_last=False)
