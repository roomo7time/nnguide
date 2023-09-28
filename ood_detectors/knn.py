import torch
from typing import Dict

from ood_detectors.interface import OODDetector
from ood_detectors.assets import knn_score

class KNNOODDetector(OODDetector):

    def setup(self, args, train_model_outputs, train_labels):
        feas_train = train_model_outputs['feas']
        
        try:
            self.knn_k = args.detector['knn_k']
        except:
            self.knn_k = 10

        self.feas_train = feas_train

    def infer(self, model_outputs: Dict):

        feas = model_outputs['feas']

        scores = knn_score(self.feas_train, feas, k=self.knn_k, min=True)
        scores = torch.from_numpy(scores).to(feas.device)
        return scores