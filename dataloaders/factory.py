import torchvision
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10, CIFAR100, SVHN
import imghdr
from natsort import natsorted

from PIL import Image
import numpy as np
import os


class ImageNetID(Dataset):
    def __init__(self, data_root_path, data_name, transform, train=False):

        if data_name == 'imagenet1k':
            if train:
                fold_name = 'imagenet1k/imagenet/train'
            else:
                fold_name = 'imagenet1k/imagenet/val'

            dataset_path = f"{data_root_path}/{fold_name}"

            _dataset = ImageFolder(root=dataset_path)
            _samples = [sample[0] for sample in _dataset.samples]

            self.targets = np.array(_dataset.targets, dtype=np.int64).tolist()

        elif data_name in ['imagenet1k-v2-a', 'imagenet1k-v2-b', 'imagenet1k-v2-c']:

            if data_name == 'imagenet1k-v2-a':
                fold_name = 'imagenet1k-v2/imagenetv2-threshold0.7-format-val'
            elif data_name == 'imagenet1k-v2-b':
                fold_name = 'imagenet1k-v2/imagenetv2-matched-frequency-format-val'
            elif data_name == 'imagenet1k-v2-c':
                fold_name = 'imagenet1k-v2/imagenetv2-top-images-format-val'

            dataset_path = f"{data_root_path}/{fold_name}"

            _samples = []
            self.targets = []
            for k in range(1000):
                _filenames = os.listdir(f"{dataset_path}/{k}")
                _samples += [f"{dataset_path}/{k}/{filename}" for filename in _filenames]
                self.targets += [k] * len(_filenames)

        else:
            raise ValueError('fold is incorrectly given')

        self.data = _samples
        self.transform = transform

        self.reject_targets = np.zeros_like(self.targets).tolist()

    def __len__(self):
        return len(self.reject_targets)

    def __getitem__(self, index):
        img, reject_target, target = self.data[index], self.reject_targets[index], self.targets[index]
        img = Image.open(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        out = self.transform(img)
        return out, reject_target, target


class ImageNetOOD(Dataset):
    def __init__(self, data_root_path, data_name, transform):

        assert data_name in ['inaturalist', 'sun', 'places', 'textures', 'openimage-o']

        dataset_paths = {
            'inaturalist': data_root_path + '/inaturalist/iNaturalist/images',
            'sun': data_root_path + '/sun/SUN/images',
            'places': data_root_path + '/places/Places/images',
            'textures': data_root_path + '/textures/dtd/images',
            'openimage-o': data_root_path + '/openimage-o/images'
        }

        if data_name in ['textures', 'imagenet-o']:
            _dataset = ImageFolder(root=dataset_paths[data_name])
            _data = np.array([sample[0] for sample in _dataset.samples])
        elif data_name in ['inaturalist', 'sun', 'places', 'openimage-o', 'species']:
            _data = []
            for _file_path in natsorted(os.listdir(dataset_paths[data_name])):
                file_path = dataset_paths[data_name] + '/' + _file_path
                if imghdr.what(file_path) in ['jpeg', 'png']:
                    _data.append(file_path)
            _data = np.array(_data)
        else:
            raise NotImplementedError()

        self.data = _data.tolist()
        self.reject_targets = np.ones(len(self.data), dtype=np.int64)
        self.targets = np.ones_like(self.reject_targets) * 1000  # should be 30 since there are only 30 known classes in ImageNet30

        self.reject_targets = self.reject_targets.tolist()
        self.targets = self.targets.tolist()

        self.transform = transform

    def __len__(self):
        return len(self.reject_targets)

    def __getitem__(self, index):
        img, reject_target, target = self.data[index], self.reject_targets[index], self.targets[index]
        img = Image.open(img)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        out = self.transform(img)
        return out, reject_target, target

class CIFARData(Dataset):
    def __init__(self, data_root_path, data_name, transform, train=True):
        if data_name == "cifar10":
            _dataset = CIFAR10(root=f"{data_root_path}/cifar10", train=train, download=True)
        elif data_name == "cifar100":
            _dataset = CIFAR100(root=f"{data_root_path}/cifar100", train=train, download=True)
        else:
            raise ValueError("Invalid dataset name")
        
        # Extracting samples and targets
        self.data = [transforms.ToPILImage()(image) for image in _dataset.data]
        self.targets = _dataset.targets

        # Since CIFAR datasets don't have reject_targets in the provided scheme
        self.reject_targets = [0] * len(self.targets)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform:
            img = self.transform(img)
        return img, self.reject_targets[index], self.targets[index]
    
class SVHNData(Dataset):
    def __init__(self, data_root_path, transform, train=True):

        if train:
            split = 'train'
        else:
            split = 'test'

        _dataset = SVHN(root=data_root_path, split=split, download=True)
        
        # Convert the numpy arrays to PIL Images
        self.data = [transforms.ToPILImage()(image) for image in _dataset.data.transpose((0, 2, 3, 1))]
        self.targets = _dataset.labels.tolist()

        # Since SVHN dataset doesn't have reject_targets in the provided scheme
        self.reject_targets = [0] * len(self.targets)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform:
            img = self.transform(img)
        return img, self.reject_targets[index], self.targets[index]

def get_dataset(data_root_path, data_name, transform, 
                train=True, ood=False):
    
    if data_name in ['cifar10', 'cifar100']:
        return CIFARData(data_root_path, data_name, transform, train)
    if data_name in ['svhn']:
        return SVHNData(data_root_path, transform, train)

    if ood:
        return ImageNetOOD(data_root_path, data_name, transform)

    dataset = ImageNetID(data_root_path, data_name, transform, train)

    return dataset

def get_train_dataloader(data_root_path, data_name, batch_size, transform, num_workers=0, bankset_ratio=0.01, shuffle=False):
    dataset = get_dataset(data_root_path, data_name, transform, train=True, ood=False)

    if bankset_ratio < 1.:
        subsample(dataset, alpha=bankset_ratio)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader

def get_id_dataloader(data_root_path, data_name, batch_size, transform, num_workers=0):
    dataset = get_dataset(data_root_path, data_name, transform, train=False, ood=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader

def get_ood_dataloader(data_root_path, data_name, batch_size, transform, num_workers=0):
    dataset = get_dataset(data_root_path, data_name, transform, train=False, ood=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dataloader

def subsample(dataset, alpha=0.01, shuffle=True):
    N = len(dataset)
    n = int(N*alpha)
    idxes = np.arange(N)
    if shuffle:
        np.random.shuffle(idxes)
    # dataset.data = np.array(dataset.data)[idxes][:n]
    # dataset.targets = np.array(dataset.targets)[idxes][:n]
    # dataset.reject_targets = np.array(dataset.reject_targets)[idxes][:n]

    dataset.data = [dataset.data[i] for i in idxes[:n]]
    dataset.targets = [dataset.targets[i] for i in idxes[:n]]
    dataset.reject_targets = [dataset.reject_targets[i] for i in idxes[:n]]









