import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

from model_engines.interface import ModelEngine
from dataloaders.factory import get_train_dataloader

class ViTModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = ViT()           
        self._data_transform = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()

        self._model.to(self._device)
        self._model.eval()
    
    def get_data_transform(self):
        return self._data_transform
    
    def set_dataloaders(self):
        self._train_dataloader = get_train_dataloader(self._data_root_path, 
                                                      self._train_data_name,
                                                      self._batch_size, 
                                                      self._data_transform,
                                                      num_workers=self._num_workers)
    
    def load_saved_model(self, saved_model):
        pass

    def train_model(self):
        pass

    def get_train_dataloader(self):
        return self._train_dataloader

    def infer(self, x):
        x = x.to(self._device)
        with torch.no_grad():
            raw_feas, logits = self._model(x)
            feas = F.normalize(raw_feas, dim=1)
        return {"feas": feas, "logits": logits}

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

