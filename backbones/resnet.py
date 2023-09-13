import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.fc = self.encoder.fc
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        feas = self.encoder(x)
        logits = self.fc(feas)
        return feas, logits
