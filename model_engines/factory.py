
from model_engines.interface import ModelEngine

def create_model_engine(args) -> ModelEngine:
    assert hasattr(args, 'model')
    assert hasattr(args, 'train_data_name')

    if args.train_data_name == "imagenet1k":
        if args.model == "resnet50-supcon":
            from model_engines.resnet_supcon import ResNetSupConModelEngine
            model_engine = ResNetSupConModelEngine()
        else:
            raise NotImplementedError()
    
    else:
        raise NotImplementedError()
    
    return model_engine