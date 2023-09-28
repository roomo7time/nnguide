import torch
from typing import Dict

from ood_detectors.interface import OODDetector

class EnergyOODDetector(OODDetector):

    def setup(self, args, train_model_outputs, train_labels):
        pass

    def infer(self, model_outputs):
        assert "logits" in model_outputs
        logits = model_outputs["logits"]
        return torch.logsumexp(logits, dim=1)