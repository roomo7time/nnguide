
import torch
import torch.nn.functional as F

from model_engines.interface import ModelEngine
from model_engines.assets import extract_features
from dataloaders.factory import get_train_dataloader, get_id_dataloader, get_ood_dataloader

class ResNet50SupConModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = ResNetSupCon()
        state_dict = torch.load('./pretrained_models/resnet50-supcon.pt', map_location='cpu')['model_state_dict']
        msg = self._model.load_state_dict(state_dict, strict=False)            
        self._data_transform = DATA_TRANSFORM

        self._model.to(self._device)
        self._model.eval()
    
    def set_dataloaders(self):
        self._dataloaders = {}
        self._dataloaders['train'] = get_train_dataloader(self._data_root_path, 
                                                         self._train_data_name,
                                                         self._batch_size, 
                                                         self._data_transform,
                                                         num_workers=self._num_workers)

        self._dataloaders['id'] = get_id_dataloader(self._data_root_path, 
                                                         self._id_data_name,
                                                         self._batch_size, 
                                                         self._data_transform,
                                                         num_workers=self._num_workers)
        self._dataloaders['ood'] = get_ood_dataloader(self._data_root_path, 
                                                         self._ood_data_name,
                                                         self._batch_size, 
                                                         self._data_transform,
                                                         num_workers=self._num_workers)

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