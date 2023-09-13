import torch
import torchvision.transforms as transforms
from torchvision.models import ViT_B_16_Weights, RegNet_Y_16GF_Weights, MobileNet_V2_Weights

from backbones.resnet_supcon import ResNetSupCon
from backbones.vit import ViT
from backbones.regnet import RegNet
from backbones.mobilenet import MobileNet
from backbones.resnet import ResNet

from backbones.verify_model_interface import verify_model_interface

def load_model(arch_name, data_name):
    if data_name in ['ood-imagenet1k', 'ood-imagenet1k-v2-a', 'ood-imagenet1k-v2-b', 'ood-imagenet1k-v2-c']:
        num_classes = 1000

        if arch_name == 'resnet50-supcon':
            model = ResNetSupCon(num_classes=num_classes)
            state_dict = torch.load('./pretrained_models/resnet50-supcon.pt', map_location='cpu')['model_state_dict']
            msg = model.load_state_dict(state_dict, strict=False)            
            data_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
            ])
        elif arch_name == 'vit':
            model = ViT(model_name='vit')
            data_transform = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        elif arch_name == 'regnet':
            model = RegNet(model_name=arch_name)
            data_transform = RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1.transforms()
        elif arch_name == 'mobilenet':
            model = MobileNet()
            data_transform = MobileNet_V2_Weights.IMAGENET1K_V2.transforms()
        elif arch_name == 'resnet50':
            model = ResNet()
            data_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            raise NotImplementedError()
    else:
        raise NotImplementedError()

    verify_model_interface(model)

    return model, data_transform