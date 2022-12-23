import os.path
from typing import Any, Callable, List, Optional

from torch.utils.data import Dataset

from nhelper.behavior import Behavior
from nhelper.types import BehaviorType
from .testpack import TestPack


class PyTorchTestPack(Dataset):
    def __init__(self, capabilities: List[str], names: List[str], test_types: List[BehaviorType], texts: List[str],
                 labels: List[Any], processor: Optional[Callable] = None):
        """
        :param capabilities:
        :param names:
        :param test_types:
        :param texts:
        :param labels:
        :param processor:
        :return:
        """
        assert len(capabilities) == len(names) == len(test_types) == len(texts) == len(labels)
        self.capabilities = capabilities
        self.names = names
        self.test_types = test_types
        self.texts = texts
        self.labels = labels
        self.processor = processor

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.processor is not None:
            return self.processor(**{
                "capability": self.capabilities[idx],
                "name": self.names[idx],
                "test_type": self.test_types[idx],
                "text": self.texts[idx],
                "labels": self.labels[idx]
            })
        return {
            "capability": self.capabilities[idx],
            "name": self.names[idx],
            "test_type": self.test_types[idx],
            "text": self.texts[idx],
            "labels": self.labels[idx]
        }

    @classmethod
    def from_testpack(cls, testpack: TestPack, processor: Callable = None):
        """Constructs a PyTorch Dataset from a TestPack"""
        capabilities, names, test_types, texts, all_labels = [], [], [], [], []

        behaviors = list(testpack.behaviors)
        for behavior in behaviors:
            for sample, labels in zip(behavior.samples, behavior.labels):
                capabilities.append(behavior.capability)
                names.append(behavior.name)
                test_types.append(behavior.test_type.value)
                texts.append(sample)
                all_labels.append(labels)

        return cls(capabilities, names, test_types, texts, all_labels, processor)

    @classmethod
    def from_saved_behaviors(cls, folder_path: str, processor: Callable = None):
        """"""
        assert os.path.isdir(folder_path), "Please provide a path to a folder."

        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pkl")]

        capabilities, names, test_types, texts, all_labels = [], [], [], [], []
        for f in files:
            behavior = Behavior.from_file(f)
            for sample, labels in zip(behavior.samples, behavior.labels):
                capabilities.append(behavior.capability)
                names.append(behavior.name)
                test_types.append(behavior.test_type.value)
                texts.append(sample)
                all_labels.append(labels)
        return cls(capabilities, names, test_types, texts, all_labels, processor)
