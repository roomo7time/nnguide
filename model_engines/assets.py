import torch
from tqdm import tqdm
import torch.nn.functional as F
import re

def extract_features(model, dataloader, device):
    model.to(device)
    model.eval()

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

    return {"feas": feas, "logits": logits, "labels": labels}

