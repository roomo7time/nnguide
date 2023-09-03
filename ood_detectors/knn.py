import torch
from typing import Dict

from ood_detectors.interface import OODDetector
from ood_detectors.assets import knn_score

class KNNOODDetector(OODDetector):

    def setup(self, feas_train, logits_train, labels_train=None, hyperparam: Dict = None):
        
        try:
            self.knn_k = hyperparam['knn_k']
        except:
            self.knn_k = 10

        self.feas_train = feas_train

    def infer(self, feas, logits):

        scores = knn_score(self.feas_train, feas, k=self.knn_k, min=True)
        scores = torch.from_numpy(scores).to(feas.device)
        return scores