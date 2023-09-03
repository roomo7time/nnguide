import torch
from typing import Dict
import torch.nn.functional as F

from ood_detectors.interface import OODDetector

class MSPOODDetector(OODDetector):

    def setup(self, feas_train, logits_train, labels_train=None, hyperparam: Dict = None):
        pass

    def infer(self, feas, logits):
        return F.softmax(logits, dim=1).max(dim=1)[0]