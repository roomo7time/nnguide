
import torch

from model_engines.interface import ModelEngine
from model_engines.assets import extract_features

from dataloaders.datasets_large import get_dataloaders

class ResNet50SupConModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = ResNetSupCon()
        state_dict = torch.load('./pretrained_models/resnet50-supcon.pt', map_location='cpu')['model_state_dict']
        msg = self._model.load_state_dict(state_dict, strict=False)            
        self._data_transform = DATA_TRANSFORM

        self._model.eval()
    
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

import torch.nn as nn
from backbones.resnet50_supcon import model_dict
class ResNetSupCon(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, num_classes=1000, train=False):
        super(ResNetSupCon, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=False),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        with torch.no_grad():
            rep = self.encoder(x)
        logits = self.fc(rep)
        return rep, logits

import torchvision.transforms as transforms
DATA_TRANSFORM = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]),
        ])