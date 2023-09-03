import torch
from tqdm import tqdm
import torch.nn.functional as F
import re

def extract_features(model, dataloader, device):

    feas = [[]] * len(dataloader)
    logits = [[]] * len(dataloader)
    labels = [[]] * len(dataloader)
    
    for i, labeled_data in tqdm(enumerate(dataloader), desc="Extracting features"):
        _x = labeled_data[0].to(device)
        _y = labeled_data[2]

        with torch.no_grad():
            _rawfeas, _logits = model(_x)
        _feas = F.normalize(_rawfeas, dim=1)

        feas[i] = _feas.cpu()
        logits[i] = _logits.cpu()
        labels[i] = _y.cpu()
    
    feas = torch.cat(feas, dim=0)
    logits = torch.cat(logits, dim=0)
    labels = torch.cat(labels, dim=0)

    print(f"Successfully extracted features")

    return feas, logits, labels

def load_features(save_dir_path, name=None):
    print(f"Loading features - fold: {name}")
    assert re.fullmatch(r'^(train|id|ood-\d+)$', name)
    tensor_dict = torch.load(f"{save_dir_path}/{name}.pt")
    print(f"Successfully loaded features - fold: {name}")
    return tensor_dict['feas'], tensor_dict['logits'], tensor_dict['labels']

def save_features(tensor_dict, save_dir_path, name=None):
    assert re.fullmatch(r'^(train|id|ood-\d+)$', name)
    print(f"Saving features - fold: {name}")
    torch.save(tensor_dict, f"{save_dir_path}/{name}.pt")
    print(f"Successfully saved features - fold: {name}")