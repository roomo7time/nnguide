import os
import json
import numpy as np
from prettytable import PrettyTable
import getpass
import yaml
from datetime import datetime
import pickle

'''
Result Save Utils
'''
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def import_yaml_config(args, yaml_path):
    # args: namespace instance
    with open(yaml_path, 'r') as stream:
        conf_yaml = yaml.safe_load(stream)

    for k, v in conf_yaml.items():
        args.__setattr__(k, v)

    return args

def save_dict(dict_obj, dict_path):
    with open(dict_path, 'wb') as f:
        pickle.dump(dict_obj, f)

def load_dict(dict_path):
    with open(dict_path, 'rb') as f:
        return pickle.load(f)


def dict2csv(dict, filename):
    ptable = dict2ptable(dict)
    print(ptable)
    ptable2csv(ptable, filename)


def dict2ptable(dict):
    # dict contains scalar values as its elements
    table_header = list(dict.keys())
    result_table = PrettyTable(table_header, align='r', float_format='.4')
    result_table.add_row(list(dict.values()))
    return result_table

def ptable2csv(table, filename, headers=True):
    """Save PrettyTable results to a CSV file.

    Adapted from @AdamSmith https://stackoverflow.com/questions/32128226

    :param PrettyTable table: Table object to get data from.
    :param str filename: Filepath for the output CSV.
    :param bool headers: Whether to include the header row in the CSV.
    :return: None
    """
    raw = table.get_string()
    data = [tuple(filter(None, map(str.strip, splitline)))
            for line in raw.splitlines()
            for splitline in [line.split('|')] if len(splitline) > 1]
    if table.title is not None:
        data = data[1:]
    if not headers:
        data = data[1:]
    with open(filename, 'w') as f:
        for d in data:
            f.write('{}\n'.format(','.join(d)))



def get_paths(args, head='./logs/imagenet'):

    args.log_dir_path = f"{head}/{args.config_name}/{args.data_name}"
    # now_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # args.tb_dir_path = './logs/tb_log/%s/%s/split-%d/%s' % (args.config_name, args.data_name, args.split_idx, now_time)

    mkdir(args.log_dir_path)
    # mkdir(args.tb_dir_path)

    return args


def put_header_dict(dict, head):
    _dict = dict
    dict = {}
    for key in _dict.keys():
        dict[f'{head}_{key}'] = _dict[key]
    return dict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

