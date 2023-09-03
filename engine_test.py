import torch
from tqdm import tqdm





# TODO: debug
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn import metrics




def compute_oscr(x1, x2, pred, labels):

    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate
    """

    x1, x2 = -x1, -x2

    # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    # pred = np.argmax(pred_k, axis=1)

    correct = (pred == labels)
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1:].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)

    OSCR = 0

    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0

        OSCR = OSCR + h * w

    # print(f'OSCR: {OSCR}')

    return OSCR



import faiss
from copy import deepcopy
def knn_score(bankfeas, queryfeas, k=10, min=False):

    bankfeas = deepcopy(np.array(bankfeas))
    queryfeas = deepcopy(np.array(queryfeas))

    index = faiss.IndexFlatIP(bankfeas.shape[-1])
    index.add(bankfeas)
    D, I = index.search(queryfeas, k)

    if min:
        scores = np.array(D.min(axis=1))
    else:
        scores = np.array(D.mean(axis=1))

    return scores

def knn_score_ablation(bankfeas, queryfeas):

    bankfeas = deepcopy(np.array(bankfeas))
    queryfeas = deepcopy(np.array(queryfeas))

    # faiss.normalize_L2(bankfeas)
    # faiss.normalize_L2(queryfeas)

    index = faiss.IndexFlatIP(bankfeas.shape[-1])
    index.add(bankfeas)
    D, I = index.search(queryfeas, 5000)

    scores = {}

    for k_knn in tqdm([1, 10, 20, 50, 100, 200, 500, 1000, 3000, 5000]):
        scores[k_knn] = D[:, :k_knn].mean(axis=1)

    return scores

def conf_knn_score(bankfeas, bankconfs, queryfeas, k=10, min=False):

    bankfeas = deepcopy(np.array(bankfeas))
    queryfeas = deepcopy(np.array(queryfeas))

    index = faiss.IndexFlatIP(bankfeas.shape[-1])
    index.add(bankfeas*bankconfs)
    D, I = index.search(queryfeas, k)

    D = D / np.take_along_axis(bankconfs, I, axis=0)

    # S = np.matmul(queryfeas, bankfeas.T)
    # D = np.take_along_axis(S, I, axis=1)
    if min:
        scores = np.array(D.min(axis=1))
    else:
        scores = np.array(D.mean(axis=1))

    return scores

class ReAct(torch.nn.Module):
    def __init__(self, c=1.0):
        super(ReAct, self).__init__()
        self.c = c

    def forward(self, x):
        return x.clip(max=self.c)


def compute_ood_performances(labels, scores):
    # labels: 0 = OOD, 1 = ID
    # scores: it is anomality score (the higher the score, the more anomalous)

    # auroc
    fpr, tpr, _ = metrics.roc_curve(labels, scores, drop_intermediate=False)
    auroc = metrics.auc(fpr, tpr)

    # tnr at tpr 95
    idx_tpr95 = np.abs(tpr - .95).argmin()
    fpr_at_tpr95 = fpr[idx_tpr95]

    # dtacc (detection accuracy)
    dtacc = .5 * (tpr + 1. - fpr).max()

    # auprc (in and out)
    # ref. https://github.com/izikgo/AnomalyDetectionTransformations/blob/master/utils.py#L91
    auprc_in = metrics.average_precision_score(y_true=labels, y_score=scores)
    auprc_out = metrics.average_precision_score(y_true=labels, y_score=-scores, pos_label=0) # equivalent to average_precision_score(y_true=1-labels, y_score=-scores)

    return auroc, fpr_at_tpr95, dtacc, auprc_in, auprc_out


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
        self.cov = np.zeros(shape=(self.num_clusters, dim, dim))

        for k in tqdm(range(self.num_clusters)):
            X_k = np.array(X[y == k])

            self.center[k] = np.mean(X_k, axis=0)
            self.cov[k] = np.cov(X_k.T, bias=True)

        if supervised:
            shared_cov = self.cov.mean(axis=0)
            self.shared_icov = np.linalg.pinv(shared_cov)
        else:
            self.icov = np.zeros(shape=(self.num_clusters, dim, dim))
            self.shared_icov = None
            for k in tqdm(range(self.num_clusters)):
                self.icov[k] = np.linalg.pinv(self.cov[k])

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



