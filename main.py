

import random
import torch
import argparse
import numpy as np

from tabulate import tabulate

from misc.utils_python import mkdir, import_yaml_config

from model_engines.factory import create_model_engine
from model_engines.interface import verify_model_outputs
from ood_detectors.factory import create_ood_detector

from eval_assets import save_performance

def verify_args(args):
    if not hasattr(args, 'seed'):
        args.seed = 0
    assert hasattr(args, 'model')
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
    parser.add_argument('--train_data_name', '-td', type=str,  
                        default='imagenet1k', 
                        choices=['imagenet1k'],
                        help='The data name for the in-distribution')
    parser.add_argument('--id_data_name', '-id', type=str,  
                        default='imagenet1k', 
                        choices=['imagenet1k', 
                                 'imagenet1k-v2-a', 
                                 'imagenet1k-v2-b', 
                                 'imagenet1k-v2-c'],
                        help='The data name for the in-distribution')
    parser.add_argument('--ood_data_name', '-ood', type=str, 
                        default='inaturalist', 
                        choices=['inaturalist', 'sun', 'places', 'textures', 'openimage-o']
                        )
    
    parser.add_argument("--ood_detectors", type=str, nargs='+', 
                        # default=['energy', 'nnguide', 'msp', 'maxlogit', 'vim', 'ssd', 'mahalanobis', 'knn'], 
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

    args.log_dir_path = f"./logs/{args.config_name}/{args.train_data_name}/{args.id_data_name}"
    args.train_save_dir_path = f"{args.save_root_path}/{args.config_name}/{args.train_data_name}"
    args.id_save_dir_path = f"{args.save_root_path}/{args.config_name}/{args.id_data_name}"
    args.ood_save_dir_path = f"{args.save_root_path}/{args.config_name}/{args.ood_data_name}"

    args = verify_args(args)

    mkdir(args.log_dir_path)
    mkdir(args.train_save_dir_path)
    mkdir(args.id_save_dir_path)
    mkdir(args.ood_save_dir_path)

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

    save_performance(scores_set, labels, accs, f"{args.log_dir_path}/ood-{args.ood_data_name}.csv")

def infer(args, ood_detector_name: str):
    
    print(f"Inferencing - OOD detector: {ood_detector_name}")

    model_engine = create_model_engine(args)
    model_engine.set_model(args)
    model_engine.set_dataloaders()
    model_engine.train_model()

    all_model_outputs = {}
    all_model_outputs['train'], all_model_outputs['id'], all_model_outputs['ood'] \
        = model_engine.get_model_outputs()

    for _, _model_outputs in all_model_outputs.items():
        assert verify_model_outputs(_model_outputs)
    
    ood_detector = create_ood_detector(ood_detector_name)

    if hasattr(args, ood_detector_name):
        hyperparam = getattr(args, ood_detector_name)
    else:
        hyperparam = None
    
    ood_detector.setup(all_model_outputs['train'], hyperparam)

    _scores = {}
    for fold in ['id', 'ood']:
        print(f"Inferring scores - detector: {ood_detector_name} fold: {fold}")
        _scores[fold] = ood_detector.infer(all_model_outputs[fold])
    
    scores = torch.cat([_scores['id'], _scores['ood']], dim=0).numpy()

    labels = {}
    labels['id'] = all_model_outputs['id']['labels']
    labels['ood'] = all_model_outputs['ood']['labels']
    id_logits = all_model_outputs['id']['logits']
    detection_labels = torch.cat([torch.ones_like(labels['id']), torch.zeros_like(labels['ood'])], dim=0).numpy()
    
    pred_id = torch.max(id_logits, dim=-1)[1]
    acc = (pred_id == labels['id']).float().mean().numpy()

    return scores, detection_labels, acc

if __name__ == '__main__':
    main()