"""
data_loader/data_loaders.py
─────────────────────────────────────────────────────────────────
DataLoader wrapper for UKBBData.

Exposes:
  .dataset  — the underlying UKBBData (for .global_mask, .train_smri_mean)
  .loader   — torch.utils.data.DataLoader ready for training
"""

from torch.utils.data import DataLoader
from .datasets import UKBBData


class UKBB:
    """
    Parameters
    ----------
    split        : 'train' | 'valid' | 'test'
    batch_size   : samples per batch
    num_subjects : cap on number of subjects (None = all)
    num_workers  : DataLoader worker processes
    shuffle      : default True for train, False otherwise
    """

    def __init__(self,
                 split:        str,
                 batch_size:   int  = 4,
                 num_subjects: int  = None,
                 num_workers:  int  = 4,
                 shuffle:      bool = None):

        self.dataset = UKBBData(split=split, num_subjects=num_subjects)

        if shuffle is None:
            shuffle = (split == 'train')

        loader_kwargs = dict(
            batch_size         = batch_size,
            shuffle            = shuffle,
            num_workers        = num_workers,
            persistent_workers = (num_workers > 0),
            pin_memory         = True,   # speed up CPU→GPU transfer
        )
        # prefetch_factor is only valid when num_workers > 0
        if num_workers > 0:
            loader_kwargs['prefetch_factor'] = 2

        self.loader = DataLoader(self.dataset, **loader_kwargs)

    def __len__(self) -> int:
        return len(self.dataset)
