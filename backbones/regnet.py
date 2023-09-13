import torch.nn as nn
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights

class RegNet(nn.Module):
    def __init__(self, model_name='regnet'):
        super(RegNet, self).__init__()

        if model_name == 'regnet':
            self.encoder = regnet_y_16gf(weights=RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1)

        self.fc = self.encoder.fc
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        feas = x

        logits = self.fc(x)

        return feas, logits

if __name__ == '__main__':
    model = RegNet('regnetß')
    print()