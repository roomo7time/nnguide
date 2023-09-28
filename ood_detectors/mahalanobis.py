
import torch
from typing import Dict

from ood_detectors.interface import OODDetector
from ood_detectors.assets import Mahalanobis

class MahalanobisOODDetector(OODDetector):

    def setup(self, args, train_model_outputs: Dict):
        feas_train = train_model_outputs['feas']
        labels_train = train_model_outputs['labels']
        self.mahalanobis = Mahalanobis()
        self.mahalanobis.fit(feas_train, labels_train)

    def infer(self, model_outputs: Dict):

        feas = model_outputs['feas']
        device = feas.device
        return torch.from_numpy(self.mahalanobis.score(feas)).to(device)