
import torch
from abc import ABC, abstractmethod
from typing import Dict

class OODDetector(ABC):

    @abstractmethod
    def setup(self, 
              feas_train: torch.Tensor, 
              logits_train: torch.Tensor, 
              labels_train: torch.Tensor = None, 
              hyperparam: Dict = None):
        pass

    @abstractmethod
    def infer(self, feas, logits):
        pass