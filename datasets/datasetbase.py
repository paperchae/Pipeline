from typing import Any, Callable, Optional, Tuple
from torch.utils.data import Dataset
import torch


class BaseDataset(Dataset):
    def __init__(
            self, config: Any,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            device: torch.device = 'cpu') -> None:
        self.cfg = config
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

        self.data = None
        self.target = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.data[idx]
        target = self.target[idx]

        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return data.to(self.device), target.to(self.device)


class Compose:
    def __init__(self, transforms: list) -> None:
        """
        Args:
            transforms (list): list of transforms to compose.
        Returns:
            Composed transforms which sequentially perform a list of transforms.
        """
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x
