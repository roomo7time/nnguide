import torch
from typing import Dict

from ood_detectors.interface import OODDetector

class EnergyOODDetector(OODDetector):

    def setup(self, feas_train, logits_train, labels_train=None, hyperparam: Dict = None):
        pass

    def infer(self, feas, logits):
        return torch.logsumexp(logits, dim=1)