import torch
from typing import Dict
import torch.nn.functional as F

from ood_detectors.interface import OODDetector

class MaxLogitOODDetector(OODDetector):

    def setup(self, args, train_model_outputs: Dict):
        pass

    def infer(self, model_outputs: Dict):

        logits = model_outputs['logits']
        return logits.max(dim=1)[0]