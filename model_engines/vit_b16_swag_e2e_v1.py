import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from model_engines.interface import ModelEngine
from model_engines.assets import extract_features

from dataloaders.factory import get_dataloaders

class ViTModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = ViT()           
        self._data_transform = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()

        if hasattr(args, 'react_percentile'):
            self._react_percentile = args.react_percentile
            self._train_save_dir_path = args.train_save_dir_path
    
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
    def load_saved_model(self):
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

