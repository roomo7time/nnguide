import torch
from typing import Dict
import torch.nn.functional as F

from ood_detectors.interface import OODDetector

class KLOODDetector(OODDetector):

    def setup(self, args, train_model_outputs):
        pass

    def infer(self, model_outputs: Dict):

        logits = model_outputs['logits']
        return F.cross_entropy(logits, torch.ones_like(logits)/logits.shape[-1], reduction='none')