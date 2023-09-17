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


import torchvision.transforms as transforms
DATA_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])