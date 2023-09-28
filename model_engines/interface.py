from abc import ABC, abstractmethod
from typing import Dict, Tuple
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader

# The model engine serves as an adapter to the ood detector
class ModelEngine(ABC):

    @abstractmethod
    def set_model(self, args):
        """
        Set up the model architecture and configurations.
        """
        self._folds = ['train', 'id', 'ood']
        
        self._data_root_path = args.data_root_path
        
        self._train_data_name = args.train_data_name

        self._batch_size = args.batch_size
        self._num_workers = args.num_workers
        self._device = args.device
        pass
    
    def get_data_transform(self) -> transforms.Compose:
        pass

    @abstractmethod
    def set_dataloaders(self):
        pass
    
    @abstractmethod
    def load_saved_model(self, saved_model: nn.Module):
        pass

    @abstractmethod
    def train_model(self) -> nn.Module:
        pass

    @abstractmethod
    def get_train_model_outputs(self) -> Dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def infer(self) -> Dict[str, torch.Tensor]:
        pass

def verify_model_outputs(model_outputs: Tuple) -> bool:

    output_types = ['feas', 'logits']
    try:
        for output_type in output_types:
            output_type in model_outputs
    except:
        return False

    return True

from tqdm import tqdm
def get_model_outputs(dataloader: DataLoader, infer: ModelEngine.infer):

    for i, data in tqdm(enumerate(dataloader), desc="Getting model outputs..."):

        x = data[0]
        y = data[2]

        output_dict = infer(x)

        if i == 0:
            labels = [[]]*len(dataloader)
            model_outputs = {}
            for key in output_dict:
                model_outputs[key] = [[]]*len(dataloader)
        
        labels[i] = y.cpu()
        for key in model_outputs:
            model_outputs[key][i] = output_dict[key].cpu()
        
    labels = torch.cat(labels, dim=0)
    for key in model_outputs:
        model_outputs[key] = torch.cat(model_outputs[key], dim=0)
    
    return model_outputs, labels
    
