from abc import ABC, abstractmethod
from typing import Dict, Tuple

class ModelEngine(ABC):

    @abstractmethod
    def set_model(self, args):
        """
        Set up the model architecture and configurations.
        """
        
        assert hasattr(args, 'train_save_dir_path')
        assert hasattr(args, 'id_save_dir_path')
        assert hasattr(args, 'ood_save_dir_path')
        
        self._folds = ['train', 'id', 'ood']

        assert hasattr(args, 'device')
        
        self._data_root_path = args.data_root_path
        self._data_names = {}
        self._data_names['train'] = args.train_data_name
        self._data_names['id'] = args.id_data_name
        self._data_names['ood'] = args.ood_data_name

        self._batch_size = args.batch_size
        self._num_workers = args.num_workers
        self._device = args.device
        pass
    
    def get_data_transform(self):
        pass

    @abstractmethod
    def set_dataloaders(self):
        pass
    
    @abstractmethod
    def load_saved_model(self, saved_model):
        pass

    @abstractmethod
    def train_model(self):
        pass
    
    @abstractmethod
    def get_model_outputs(self) -> Tuple[Dict, Dict, Dict]:
        pass

def verify_model_outputs(model_outputs: Tuple) -> bool:

    output_types = ['feas', 'logits', 'labels']
    try:
        for output_type in output_types:
            output_type in model_outputs
    except:
        return False

    return True
