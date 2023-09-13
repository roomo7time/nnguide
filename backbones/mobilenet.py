import torch
import torch.nn as nn
from torchvision.models import mobilenetv2, MobileNet_V2_Weights

class MobileNet(nn.Module):
    def __init__(self, ):
        super(MobileNet, self).__init__()

        self.encoder = mobilenetv2.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
        self.drop = self.encoder.classifier[0]
        self.fc = self.encoder.classifier[1]
        self.encoder.classifier = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        feas = x
        x = self.drop(x)
        logits = self.fc(x)

        return feas, logits


if __name__ == '__main__':
    model = MobileNet()
    x = torch.randn((10, 3, 224, 224))
    model(x)