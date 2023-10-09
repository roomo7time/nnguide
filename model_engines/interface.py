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
        self._id_data_name = args.id_data_name
        self._ood_data_name = args.ood_data_name

        self._batch_size = args.batch_size
        self._num_workers = args.num_workers
        self._device = args.device
        pass

    @abstractmethod
    def set_dataloaders(self):
        pass

    @abstractmethod
    def train_model(self) -> nn.Module:
        pass

    @abstractmethod
    def get_model_outputs(self) -> Dict:
        pass

def verify_model_outputs(model_outputs: Tuple) -> bool:

    output_types = ['feas', 'logits', 'labels']
    try:
        for output_type in output_types:
            output_type in model_outputs
    except:
        return False

    return True


    
