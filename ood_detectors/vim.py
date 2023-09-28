import torch
import numpy as np
from sklearn.covariance import EmpiricalCovariance

from typing import Dict


from ood_detectors.interface import OODDetector

class VIMOODDetector(OODDetector):

    # Unlike the original VIM, we use the mean of train features instead of bias. 
    # This is because sometimes the classifier may not have a bias, 
    # and PCA makes more sense with the mean than the bias.
    def setup(self, args, train_model_outputs, train_labels):
        feas_train = train_model_outputs['feas']
        logits_train = train_model_outputs['logits']
        
        device = feas_train.device

        u = feas_train.mean(dim=0, keepdim=True)
        X = feas_train - u
        
        ec = EmpiricalCovariance(assume_centered=True)
        ec.fit(X.numpy())
        cov_mat = ec.covariance_
        eig_vals, eigen_vectors = np.linalg.eig(cov_mat)
        R = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[500:]]).T)      # from the official ViM repository
        
        R = torch.from_numpy(R).to(device)

        virtual_logits_train = torch.mm(X, R).norm(dim=1)
        alpha = logits_train.max(dim=1)[0].mean() / virtual_logits_train.mean()

        self.u = u
        self.R = R
        self.alpha = alpha

    def infer(self, model_outputs: Dict):

        feas = model_outputs['feas']
        logits = model_outputs['logits']

        device = feas.device

        u = self.u.to(device)
        R = self.R.to(device)
        alpha = self.alpha.to(device)

        virtual_logits = torch.mm(feas - self.u, R).norm(dim=1) * alpha
        energies = torch.logsumexp(logits, dim=1)

        return energies - virtual_logits