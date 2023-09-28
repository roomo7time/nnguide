import torch
import torch.nn as nn
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights



from model_engines.interface import ModelEngine
from model_engines.assets import extract_features

from datasets_large import get_dataloaders

class RegNetModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = RegNet()           
        self._data_transform = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()

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
        pass

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


class RegNet(nn.Module):
    def __init__(self, model_name='regnet'):
        super(RegNet, self).__init__()

        if model_name == 'regnet':
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