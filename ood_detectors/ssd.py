
import torch
from typing import Dict

from ood_detectors.interface import OODDetector
from ood_detectors.assets import Mahalanobis

class SSDOODDetector(OODDetector):

    def setup(self, feas_train, logits_train, labels_train, hyperparam: Dict = None):
        self.ssd = Mahalanobis(num_clusters=1)
        self.ssd.fit(feas_train)

    def infer(self, feas, logits):
        device = feas.device
        return torch.from_numpy(self.ssd.score(feas)).to(device)