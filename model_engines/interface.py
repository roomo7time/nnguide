from abc import ABC, abstractmethod
from typing import Dict

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
        self._save_file_paths = {}
        self._save_file_paths['train'] = f"{args.train_save_dir_path}/model_outputs_train.pt"
        self._save_file_paths['id'] = f"{args.id_save_dir_path}/model_outputs_id.pt"
        self._save_file_paths['ood'] = f"{args.ood_save_dir_path}/model_outputs_ood.pt"

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

    @abstractmethod
    def set_dataloaders(self):
        pass
    
    @abstractmethod
    def train_model(self):
        pass
    
    @abstractmethod
    def get_model_outputs(self):
        pass
        


def verify_model_outputs(model_outputs: Dict) -> bool:
    output_types = ['feas', 'logits', 'labels']

    for output_type in output_types:
        output_type in model_outputs

    return True