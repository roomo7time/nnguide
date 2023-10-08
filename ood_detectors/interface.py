
import torch
from abc import ABC, abstractmethod
from typing import Dict

class OODDetector(ABC):

    @abstractmethod
    def setup(self, args, train_model_outputs: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def infer(self, model_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass