import torch
import torch.nn as nn
import inspect

def verify_model_interface(model):
    # Check if the model has a 'forward' method
    assert hasattr(model, 'forward'), "Model is missing the 'forward' method"
    assert hasattr(model, 'encoder'), "Model is missing the 'encoder' attribute"
    assert hasattr(model, 'fc'), "Model is missing the 'fc' attribute"
    
    # Checking that the 'encoder' and 'fc' are nn.Module objects
    assert isinstance(model, nn.Module)
    assert isinstance(model.encoder, nn.Module), "'encoder' should be an instance of nn.Module"
    assert isinstance(model.fc, nn.Module), "'fc' should be an instance of nn.Module"
