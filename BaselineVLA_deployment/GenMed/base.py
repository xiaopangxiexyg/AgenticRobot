from __future__ import annotations

from typing import Any

from torch.utils.data import Dataset


class BaseMedProvider:
    name = "base"

    def build_med(self, sample: Any) -> Any:
        raise NotImplementedError

    def infer_output_spec(self, data_summary: dict[str, Any]) -> dict[str, Any] | None:
        med_dim = self.infer_output_dim(data_summary)
        if med_dim is None:
            return None
        return {
            "kind": "vector",
            "shape": [int(med_dim)],
            "flat_dim": int(med_dim),
            "provider": self.name,
        }

    def infer_output_dim(self, data_summary: dict[str, Any]) -> int | None:
        del data_summary
        return None


class MedAugmentedDataset(Dataset):
    def __init__(self, dataset: Dataset, med_provider: BaseMedProvider) -> None:
        self.dataset = dataset
        self.med_provider = med_provider

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        sample = self.dataset[index]
        med_value = self.med_provider.build_med(sample)
        sample.intermediates["med"] = med_value
        sample.availability["med"] = med_value.new_tensor(1.0)
        sample.metadata["med_data_type"] = self.med_provider.name
        sample.metadata["med_shape"] = [int(value) for value in med_value.shape]
        return sample
