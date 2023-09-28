import torch
import torch.nn as nn
from torchvision.models import mobilenetv2, MobileNet_V2_Weights

from model_engines.interface import ModelEngine
from model_engines.assets import extract_features

from dataloaders.factory import get_dataloaders

class MobileNetModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = MobileNet()           
        self._data_transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
    
    def get_data_transform(self):
        return self._data_transform
    
    def set_dataloaders(self):
        
        self._dataloaders = {}
        self._dataloaders['train'], self._dataloaders['id'], self._dataloaders['ood'] \
            = get_dataloaders(self._data_root_path, 
                              self._data_names['train'], self._data_names['id'], self._data_names['ood'],
                              self._batch_size, 
                              self._data_transform,
                              num_workers=self._num_workers)
    
    def load_saved_model(self, saved_model):
        pass
    
    def train_model(self):
        pass

    def get_model_outputs(self):

        model_outputs = {}
        for fold in self._folds:
            model_outputs[fold] = {}

            _dataloader = self._dataloaders[fold]
            _tensor_dict = extract_features(self._model, _dataloader, self._device)
            
            model_outputs[fold]["feas"] = _tensor_dict["feas"]
            model_outputs[fold]["logits"] = _tensor_dict["logits"]
            model_outputs[fold]["labels"] = _tensor_dict["labels"]
        
        return model_outputs['train'], model_outputs['id'], model_outputs['ood']

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
