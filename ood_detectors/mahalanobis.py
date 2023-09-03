
import torch
from typing import Dict

from ood_detectors.interface import OODDetector
from ood_detectors.assets import Mahalanobis

class MahalanobisOODDetector(OODDetector):

    def setup(self, feas_train, logits_train, labels_train, hyperparam: Dict = None):
        self.mahalanobis = Mahalanobis()
        self.mahalanobis.fit(feas_train, labels_train)

    def infer(self, feas, logits):
        device = feas.device
        return torch.from_numpy(self.mahalanobis.score(feas)).to(device)