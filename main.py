

import random
import torch
import argparse
import numpy as np

from tabulate import tabulate

from datasets_large import get_test_dataloaders
from misc.utils_python import mkdir, import_yaml_config
from models.factory import load_model
from eval_assets import save_performance

def verify_args(args):
    if not hasattr(args, 'seed'):
        args.seed = 0
    assert hasattr(args, 'arch')
    return args

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-cn', type=str, 
                        default='config01',
                        help='The name of configuration')

    parser.add_argument('--gpu_idx', '-g', type=int, 
                        default=0, 
                        help='gpu idx')
    parser.add_argument('--num_workers', '-nw', type=int, 
                        default=8, 
                        help='number of workers')
    parser.add_argument('--data_name', '-d', type=str, 
                        default='ood-imagenet1k', 
                        choices=['ood-imagenet1k', 
                                 'ood-imagenet1k-v2-a', 
                                 'ood-imagenet1k-v2-b', 
                                 'ood-imagenet1k-v2-c'],
                        help='The data name for the in-distribution')
    parser.add_argument('--split_idx', '-si', type=int, 
                        default=0, 
                        choices=[0, 1, 2, 3, 4],
                        help='The index for the ood-distribution data: \
                              0: iNaturalist\
                              1: SUN \
                              2: Places \
                              3: Texture \
                              4: OpenImage-O')
    
    parser.add_argument("--ood_detectors", type=str, nargs='+', 
                        default=['energy', 'nnguide'], 
                        help="List of OOD detectors")

    parser.add_argument('--batch_size', '-bs', type=int, 
                        default=64, 
                        help='batch size for inference')

    parser.add_argument('--data_root_path', type=str, 
                        default='/home/jay/savespace/database/generic_large', 
                        help='data root path')
    parser.add_argument('--save_root_path', type=str,
                        default='./saved_model_outputs')

    args = parser.parse_args()
    args.device = torch.device('cuda:%d' % (args.gpu_idx) if torch.cuda.is_available() else 'cpu')

    args = import_yaml_config(args, head='./configs')
    args.log_dir_path = f"./logs/{args.data_name}/{args.config_name}"
    args.save_dir_path = f"{args.save_root_path}/{args.data_name}/{args.config_name}"

    args = verify_args(args)

    mkdir(args.log_dir_path)
    mkdir(args.save_dir_path)

    print(tabulate(list(vars(args).items()), headers=['arguments', 'values']))

    return args

def main():

    args = get_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    scores_set = {}
    accs = {}
    for oodd_name in args.ood_detectors:
        scores_set[oodd_name], labels, accs[oodd_name] = infer(args, oodd_name)

    save_performance(scores_set, labels, accs, f"{args.log_dir_path}/split-{args.split_idx}.csv")
    


def infer(args, ood_detector_name: str):
    
    print(f"Inferencing - OOD detector: {ood_detector_name}")

    model, data_transform = load_model(args.arch, args.data_name)
    print('pretrained model loaded')

    dataloader = {}
    dataloader['train'], dataloader['id'], dataloader['ood'] \
        = get_test_dataloaders(args.data_root_path, args.data_name, args.split_idx, args.batch_size, data_transform,
                               num_workers=args.num_workers)

    model = model.to(args.device)
    model.eval()

    '''
    apply react
    '''
    from models.apply_react import apply_react
    if hasattr(args, 'react_percentile'):
        model = apply_react(model, dataloader['train'], args.device, args.save_dir_path, args.react_percentile)

    '''
    bankset confidence construction
    '''
    # extract train features
    from models.assets import extract_features, load_features, save_features

    feas = {}
    logits = {}
    labels = {}

    folds = ['train', 'id', 'ood']

    for fold in folds:
        print(f"Preparing features - detector: {ood_detector_name} fold: {fold}")

        fold_name = fold if fold in ['train', 'id'] else f"{fold}-{args.split_idx}"

        try:
            feas[fold], logits[fold], labels[fold] = load_features(args.save_dir_path, name=fold_name)
        except:
            print(f"Failed at loading features - detector: {ood_detector_name} fold: {fold_name}")
            feas[fold], logits[fold], labels[fold] = extract_features(model, dataloader[fold], args.device)
            save_features({"feas": feas[fold],
                            "logits": logits[fold],
                            "labels": labels[fold]
                            },
                            args.save_dir_path, 
                            name=fold_name)
    
    from ood_detectors.factory import load_ood_detector
    ood_detector = load_ood_detector(ood_detector_name)

    if hasattr(args, ood_detector_name):
        hyperparam = getattr(args, ood_detector_name)
    else:
        hyperparam = None
    
    ood_detector.setup(feas['train'], logits['train'], labels['train'],
                       hyperparam=hyperparam)
    
    _scores = {}
    for fold in ['id', 'ood']:
        print(f"Inferring scores - detector: {ood_detector_name} fold: {fold}")
        _scores[fold] = ood_detector.infer(feas[fold], logits[fold])
    
    scores = torch.cat([_scores['id'], _scores['ood']], dim=0).numpy()
    detection_labels = torch.cat([torch.ones_like(labels['id']), torch.zeros_like(labels['ood'])], dim=0).numpy()
    
    pred_id = torch.max(logits['id'], dim=-1)[1]
    acc = (pred_id == labels['id']).float().mean().numpy()

    return scores, detection_labels, acc

if __name__ == '__main__':
    main()