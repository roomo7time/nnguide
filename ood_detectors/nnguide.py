import torch
from typing import Dict

from ood_detectors.interface import OODDetector
from ood_detectors.assets import knn_score

class NNGuideOODDetector(OODDetector):

    def setup(self, feas_train, logits_train, labels_train=None, hyperparam: Dict = None):
        try:
            self.knn_k = hyperparam['knn_k']
        except:
            self.knn_k = 10

        confs_train = torch.logsumexp(logits_train, dim=1)
        self.scaled_feas_train = feas_train * confs_train[:, None]

    def infer(self, feas, logits):

        confs = torch.logsumexp(logits, dim=1)
        guidances = knn_score(self.scaled_feas_train, feas, k=self.knn_k)
        scores = torch.from_numpy(guidances).to(confs.device)*confs
        return scores