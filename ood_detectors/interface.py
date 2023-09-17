
import torch
from abc import ABC, abstractmethod
from typing import Dict

class OODDetector(ABC):

    @abstractmethod
    def setup(self, 
              train_model_outputs: Dict,
              hyperparam: Dict = None):
        pass

    @abstractmethod
    def infer(self, model_outputs: Dict):
        pass