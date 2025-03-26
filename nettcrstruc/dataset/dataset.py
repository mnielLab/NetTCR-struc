from pathlib import Path
from typing import Union

import torch
import torch_geometric


class MQAInferenceDataset(torch_geometric.data.Dataset):
    """Map-style dataset for fetching torch_geometric Data objects from a directory.

    Args:
        complex_file_list: List with complex feature files.
        interface_file_list: List with interface feature files.
    """

    def __init__(
        self,
        feature_file_list: Union[Path, str],
        transform=None,
        pre_transform=None,
    ):
        super().__init__(None, transform, pre_transform)

        self.feature_file_list = feature_file_list

    def len(self) -> int:
        return len(self.feature_file_list)

    def get(self, idx) -> tuple:
        return torch.load(self.feature_file_list[idx])
