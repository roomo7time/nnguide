import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights, swin_s, Swin_S_Weights, swin_b, Swin_B_Weights



class ViT(nn.Module):
    def __init__(self, model_name='vit'):
        super(ViT, self).__init__()

        assert model_name in ['vit', 'deit', 'vit-lp']

        # define network IMAGENET1K_SWAG_E2E_V1
        if model_name == 'vit':
            self.encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        elif model_name == 'deit':
            self.encoder = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        elif model_name == 'vit-lp':
            self.encoder = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)

        self.ln = self.encoder.encoder.ln
        self.fc = self.encoder.heads.head
        self.encoder.encoder.ln = nn.Identity()
        self.encoder.heads = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        feas = x
        x = self.ln(x)

        # if self.post_ln:
        #     feas = x

        logits = self.fc(x)

        return feas, logits


class Swin(nn.Module):
    def __init__(self, arch_name='swin'):
        super(Swin, self).__init__()

        if arch_name == 'swin':
            self.encoder = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1)
        elif arch_name == 'swin-b':
            self.encoder = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)

        self.fc = self.encoder.head
        self.encoder.head = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        feas = x
        logits = self.fc(x)

        return feas, logits


if __name__ == '__main__':
    # model = ViT('vit')
    model = Swin()
    x = torch.randn((10, 3, 224, 224))
    out = model(x)