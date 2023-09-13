import torch

from backbones.resnet import ResNet
from model_engines.interface import ModelEngine, verify_model_outputs
from model_engines.assets import extract_features

class ResNetModelEngine(ModelEngine):
    def set_model(self, args, **kwargs):
        super().set_model(args, kwargs)
        self.model = ResNet()
    
    def train_model(self, args, **kwargs):
        super().train_model(args, kwargs)
        pass

    def get_model_outputs(self, args, **kwargs):
        super().get_model_outputs(args, kwargs)

        all_model_outputs = {}
        for fold in ['train', 'id', 'ood']:
            try:
                _tensor_dict = torch.load(self.save_file_path[fold])
            except:
                _dataloader = kwargs['dataloader'][fold]
                _tensor_dict = extract_features(self.model, _dataloader, args.device)
                torch.save(_tensor_dict, self.save_file_path[fold])
            
            all_model_outputs[fold]["feas"] = _tensor_dict["feas"]
            all_model_outputs[fold]["logits"] = _tensor_dict["logits"]
            all_model_outputs[fold]["labels"] = _tensor_dict["lables"]
        
            assert verify_model_outputs(all_model_outputs[fold])
        return all_model_outputs['train'], all_model_outputs['id'], all_model_outputs['ood']
