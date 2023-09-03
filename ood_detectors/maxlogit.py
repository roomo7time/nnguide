import torch
from typing import Dict
import torch.nn.functional as F

from ood_detectors.interface import OODDetector

class MaxLogitOODDetector(OODDetector):

    def setup(self, feas_train, logits_train, labels_train=None, hyperparam: Dict = None):
        pass

    def infer(self, feas, logits):
        return logits.max(dim=1)[0]