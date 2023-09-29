import numpy as np

import faiss
from copy import deepcopy
def knn_score(feas_train, feas, k=10, min=False):

    feas_train = deepcopy(np.array(feas_train))
    feas = deepcopy(np.array(feas))

    index = faiss.IndexFlatIP(feas_train.shape[-1])
    index.add(feas_train)
    D, I = index.search(feas, k)

    if min:
        scores = np.array(D.min(axis=1))
    else:
        scores = np.array(D.mean(axis=1))

    return scores


import numpy as np
from sklearn.preprocessing import normalize
from tqdm import tqdm
class Mahalanobis(object):
    def __init__(self, normalize_on=True, standardize_on=True, num_clusters=5):

        self.normalize_on = normalize_on
        self.standardize_on = standardize_on
        self.num_clusters = num_clusters  # the number of K-means clusters

    def fit(self, X, y=None):

        if y is None:
            supervised = False
        else:
            supervised = True

        X = np.array(X)
        y = np.array(y)
        dim = X.shape[1]

        self.mean = np.mean(X, axis=0, keepdims=True)
        self.std = np.std(X, axis=0, keepdims=True)

        X = self._preprocess(X)

        # clustering
        if supervised:
            self.num_clusters = len(np.unique(y))
        else:
            # link: https://github.com/inspire-group/SSD/blob/main/eval_ssdk.py
            if self.num_clusters > 1:
                kmeans = faiss.Kmeans(d=X.shape[1], k=self.num_clusters, niter=100, verbose=False, gpu=False)
                kmeans.train(np.array(X))
                y = np.array(kmeans.assign(X)[1])
            else:
                y = np.zeros(len(X))

        self.center = np.zeros(shape=(self.num_clusters, dim))
        cov = np.zeros(shape=(self.num_clusters, dim, dim))

        for k in tqdm(range(self.num_clusters)):
            X_k = np.array(X[y == k])

            self.center[k] = np.mean(X_k, axis=0)
            cov[k] = np.cov(X_k.T, bias=True)

        if supervised:
            shared_cov = cov.mean(axis=0)
            self.shared_icov = np.linalg.pinv(shared_cov)
        else:
            self.icov = np.zeros(shape=(self.num_clusters, dim, dim))
            self.shared_icov = None
            for k in tqdm(range(self.num_clusters)):
                self.icov[k] = np.linalg.pinv(cov[k])

    def score(self, X, return_distance=False):
        X = np.array(X)
        X = self._preprocess(X)

        if self.shared_icov is not None:
            M = self.shared_icov
            U = self.center
            md = (np.matmul(X, M) * X).sum(axis=1)[:, None] \
                 + ((np.matmul(U, M) * U).sum(axis=1).T)[None, :] \
                 - 2 * np.matmul(np.matmul(X, M), U.T)
        else:
            md = []
            for k in tqdm(range(self.num_clusters)):
                delta_k = X - self.center[k][None, :]
                md.append((np.matmul(delta_k, self.icov[k]) * delta_k).sum(axis=1))
            md = np.array(md).T

        out = md.min(axis=1)

        if return_distance:
            return out

        return np.exp(-(out/2048) / 2)
        # return np.exp(-out / 2)

    def _preprocess(self, X):
        if self.normalize_on:
            X = normalize(X, axis=1)    # normalize

        if self.standardize_on:
            X = (X - self.mean) / (self.std + 1e-8)     # standardize

        return X

    def _mahalanobis_score(self, x, center, icov):
        delta = x - center
        ms = (np.matmul(delta, icov) * delta).sum(axis=1)
        return np.maximum(ms, 0)
