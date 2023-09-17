import torch
from typing import Dict
import torch.nn.functional as F

from ood_detectors.interface import OODDetector

class MaxLogitOODDetector(OODDetector):

    def setup(self, train_model_outputs: Dict, hyperparam: Dict = None):
        pass

    def infer(self, model_outputs: Dict):

        logits = model_outputs['logits']
        return logits.max(dim=1)[0]