import torch
from typing import Dict
import torch.nn.functional as F

from ood_detectors.interface import OODDetector

class KLOODDetector(OODDetector):

    def setup(self, feas_train, logits_train, labels_train=None, hyperparam: Dict = None):
        pass

    def infer(self, feas, logits):
        return F.cross_entropy(logits, torch.ones_like(logits)/logits.shape[-1], reduction='none')