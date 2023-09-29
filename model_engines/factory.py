
from model_engines.interface import ModelEngine

def create_model_engine(model_name) -> ModelEngine:
    
    if model_name == "resnet50-react":
        from model_engines.resnet50_react import ResNet50ReActModelEngine as ModelEngine
    elif model_name == "resnet50-supcon":
        from model_engines.resnet50_supcon import ResNet50SupConModelEngine as ModelEngine
    elif model_name == "vit-b16-swag-e2e-v1":
        from model_engines.vit_b16_swag_e2e_v1 import ViTModelEngine as ModelEngine
    elif model_name == "regnet-y-16gf-swag-e2e-v1":
        from model_engines.regnet import RegNetModelEngine as ModelEngine
    elif model_name == "mobilenet-v2":
        from model_engines.mobilenet_v2 import MobileNetModelEngine as ModelEngine
    else:
        raise NotImplementedError()

    return ModelEngine()