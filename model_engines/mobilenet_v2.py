import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenetv2, MobileNet_V2_Weights

from model_engines.interface import ModelEngine
from model_engines.assets import extract_features

from dataloaders.factory import get_train_dataloader

class MobileNetModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = MobileNet()           
        self._data_transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
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
