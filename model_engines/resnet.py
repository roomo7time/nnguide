import torch
import torchvision.transforms as transforms

from model_engines.interface import ModelEngine
from model_engines.assets import extract_features

from datasets_large import get_dataloaders

class ResNetModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = ResNet()           
        self._data_transform = DATA_TRANSFORM

        if hasattr(args, 'react_percentile'):
            self._react_percentile = args.react_percentile
            self._train_save_dir_path = args.train_save_dir_path
    
    def set_dataloaders(self):
        
        self._dataloaders = {}
        self._dataloaders['train'], self._dataloaders['id'], self._dataloaders['ood'] \
            = get_dataloaders(self._data_root_path, 
                              self._data_names['train'], self._data_names['id'], self._data_names['ood'],
                              self._batch_size, 
                              self._data_transform,
                              num_workers=self._num_workers)
    
    def train_model(self):
        self._model = apply_react(self._model, self._dataloaders['train'], self._device, self._train_save_dir_path,
                                  self._react_percentile)

    def get_model_outputs(self):

        all_model_outputs = {}
        for fold in self._folds:
            all_model_outputs[fold] = {}
            try:
                _tensor_dict = torch.load(self._save_file_paths[fold])
            except:
                _dataloader = self._dataloaders[fold]
                _tensor_dict = extract_features(self._model, _dataloader, self._device)
                torch.save(_tensor_dict, self._save_file_paths[fold])
            
            all_model_outputs[fold]["feas"] = _tensor_dict["feas"]
            all_model_outputs[fold]["logits"] = _tensor_dict["logits"]
            all_model_outputs[fold]["labels"] = _tensor_dict["labels"]
        
        return all_model_outputs['train'], all_model_outputs['id'], all_model_outputs['ood']


import numpy as np
from tqdm import tqdm
import os
def apply_react(model, dataloader_train, device, save_dir_path, react_percentile=0.95):
    
    model.eval()
    model = model.to(device)

    os.makedirs(save_dir_path, exist_ok=True)
    react_artifact_file_path = f"{save_dir_path}/react.pt"

    try:
        react_artifacts = torch.load(react_artifact_file_path)
        c = react_artifacts['c']
    except:
        feas = [[]] * len(dataloader_train)
        for i, labeled_data in tqdm(enumerate(dataloader_train), desc=f"{apply_react.__name__}"):
            _x = labeled_data[0].to(device)

            with torch.no_grad():
                _feas, _ = model(_x)

            feas[i] = _feas.cpu()

        feas = torch.cat(feas, dim=0).numpy()
        c = np.quantile(feas, react_percentile)
        torch.save({"c": c}, react_artifact_file_path)
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