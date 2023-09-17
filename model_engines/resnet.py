import torch
import torchvision.transforms as transforms

from backbones.resnet import ResNet, DATA_TRANSFORM
from model_engines.interface import ModelEngine
from model_engines.assets import extract_features

from datasets_large import get_dataloaders

class ResNetModelEngine(ModelEngine):
    def set_model(self, args):
        super().set_model(args)
        self._model = ResNet()
        state_dict = torch.load('./pretrained_models/resnet50-supcon.pt', map_location='cpu')['model_state_dict']
        msg = self._model.load_state_dict(state_dict, strict=False)            
        self._data_transform = DATA_TRANSFORM
    
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



def apply_react(model, dataloader_train, device, save_dir_path, react_percentile=0.95):
    
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
        print(f"{(feas < c).mean().round(2)}% of the units of train features are less than {c}")

    print(f"ReAct c = {c}")
    model.encoder = torch.nn.Sequential(model.encoder, ReAct(c))

    return model