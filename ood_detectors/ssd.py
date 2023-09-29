
import torch
from typing import Dict

from ood_detectors.interface import OODDetector
from ood_detectors.assets import Mahalanobis

class SSDOODDetector(OODDetector):

    def setup(self, args, train_model_outputs, train_labels):
        feas_train = train_model_outputs['feas']
        
        self.ssd = Mahalanobis(num_clusters=1)
        self.ssd.fit(feas_train)

    def infer(self, model_outputs: Dict):

        feas = model_outputs['feas']
        
        device = feas.device
        return torch.from_numpy(self.ssd.score(feas)).to(device)