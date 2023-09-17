
from model_engines.interface import ModelEngine

def create_model_engine(args) -> ModelEngine:
    assert hasattr(args, 'model')
    assert hasattr(args, 'train_data_name')

    if args.train_data_name == "imagenet1k":
        if args.model == "resnet50-supcon":
            from model_engines.resnet_supcon import ResNetSupConModelEngine
            model_engine = ResNetSupConModelEngine()
        elif args.model == 'vit':
            from model_engines.vit import ViTModelEngine
            model_engine = ViTModelEngine()
        elif args.model == 'regnet':
            from model_engines.regnet import RegNetModelEngine
            model_engine = RegNetModelEngine()
        elif args.model == "resnet50":
            from model_engines.resnet import ResNetModelEngine
            model_engine = ResNetModelEngine()
        else:
            raise NotImplementedError()
    
    else:
        raise NotImplementedError()
    
    return model_engine