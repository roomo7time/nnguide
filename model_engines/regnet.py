import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights

from model_engines.interface import ModelEngine
from model_engines.assets import extract_features

from dataloaders.factory import get_train_dataloader

class RegNetModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        if args.model_name == "regnet-y-16gf-swag-e2e-v1":
            self._model = RegNet("regnet-y-16gf-swag-e2e-v1")           
            self._data_transform = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        else:
            raise NotImplementedError()
        
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

class RegNet(nn.Module):
    def __init__(self, model_name="regnet-y-16gf-swag-e2e-v1"):
        super(RegNet, self).__init__()

        if model_name == "regnet-y-16gf-swag-e2e-v1":
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