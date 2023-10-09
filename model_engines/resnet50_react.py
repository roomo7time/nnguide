from typing import Dict
import torch
import torchvision.transforms as transforms

from model_engines.interface import ModelEngine
from model_engines.assets import extract_features

from dataloaders.factory import get_train_dataloader, get_id_dataloader, get_ood_dataloader

class ResNet50ReActModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = ResNet()           
        self._data_transform = DATA_TRANSFORM

        self._react_percentile = args.model["react_percentile"]

        self._model.to(self._device)
        self._model.eval()
    
    def get_data_transform(self):
        return self._data_transform
    
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
        self._model = apply_react(self._model, self._dataloaders['train'], self._device, 
                                  self._react_percentile)
    
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

import numpy as np
from tqdm import tqdm
import os
def apply_react(model, dataloader_train, device, react_percentile=0.95):
    
    model.eval()
    model = model.to(device)
    
    feas = [[]] * len(dataloader_train)
    for i, labeled_data in tqdm(enumerate(dataloader_train), desc=f"{apply_react.__name__}"):
        _x = labeled_data[0].to(device)

        with torch.no_grad():
            _feas, _ = model(_x)

        feas[i] = _feas.cpu()

    feas = torch.cat(feas, dim=0).numpy()
    c = np.quantile(feas, react_percentile)
    print(f"{((feas < c).mean()*100).round(2)}% of the units of train features are less than {c}")

    print(f"ReAct c = {c}")
    model.encoder = torch.nn.Sequential(model.encoder, ReAct(c))

    return model

class ReAct(torch.nn.Module):
    def __init__(self, c=1.0):
        super(ReAct, self).__init__()
        self.c = c

    def forward(self, x):
        return x.clip(max=self.c)


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