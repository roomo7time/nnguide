

def load_ood_detector(name: str):

    if name == "energy":
        from ood_detectors.energy import EnergyOODDetector
        return EnergyOODDetector()
    elif name == "nnguide":
        from ood_detectors.nnguide import NNGuideOODDetector
        return NNGuideOODDetector()
    elif name == "vim":
        from ood_detectors.vim import VIMOODDetector
        return VIMOODDetector()
    elif name == "mahalanobis":
        from ood_detectors.mahalanobis import MahalanobisOODDetector
        return MahalanobisOODDetector()
    elif name == "ssd":
        from ood_detectors.ssd import SSDOODDetector
        return SSDOODDetector()
    elif name == "knn":
        from ood_detectors.knn import KNNOODDetector
        return KNNOODDetector()
    elif name == "msp":
        from ood_detectors.msp import MSPOODDetector
        return MSPOODDetector()
    elif name == "kl":
        from ood_detectors.kl import KLOODDetector
        return KLOODDetector()
    elif name == "maxlogit":
        from ood_detectors.maxlogit import MaxLogitOODDetector
        return MaxLogitOODDetector()
    else:
        raise NotImplementedError()