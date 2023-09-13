
import torch
import numpy as np
from tqdm import tqdm

class ReAct(torch.nn.Module):
    def __init__(self, c=1.0):
        super(ReAct, self).__init__()
        self.c = c

    def forward(self, x):
        return x.clip(max=self.c)

def apply_react(model, dataloader_train, device, save_dir_path, react_percentile=0.95):
    
    react_artifact_file_path = f"{save_dir_path}/react.pt"

    try:
        react_artifacts = torch.load(react_artifact_file_path)
        c = react_artifacts['c']
    except:
        feas = [[]] * len(dataloader_train)
        for i, labeled_data in tqdm(enumerate(dataloader_train), desc=f"{apply_react.__name__}"):
            _x = labeled_data[0].to(device)

            with torch.no_grad():
                _feas, _ = model(_x)

            feas[i] = _feas.cpu()

        feas = torch.cat(feas, dim=0).numpy()
        c = np.quantile(feas, react_percentile)
        torch.save({"c": c}, react_artifact_file_path)
        print(f"{(feas < c).mean().round(2)}% of the units of train features are less than {c}")

    print(f"ReAct c = {c}")
    model.encoder = torch.nn.Sequential(model.encoder, ReAct(c))

    return model