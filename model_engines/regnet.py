import torch
import torch.nn as nn
from torchvision.models import regnet_y_16gf, RegNet_Y_16GF_Weights



from model_engines.interface import ModelEngine
from model_engines.assets import extract_features

from dataloaders.factory import get_dataloaders

class RegNetModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        if args.model_name == "regnet-y-16gf-swag-e2e-v1":
            self._model = RegNet("regnet-y-16gf-swag-e2e-v1")           
            self._data_transform = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        else:
            raise NotImplementedError()
    
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