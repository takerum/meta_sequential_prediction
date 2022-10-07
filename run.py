import os
import argparse
import yaml
import copy
import functools
import random
import argparse
import numpy as np

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training import extensions
from utils import yaml_utils as yu


def train(config):

    torch.cuda.empty_cache()
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])

    if torch.cuda.is_available():
        device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        device = torch.device('cpu')
        gpu_index = -1

    # Dataaset
    data = yu.load_component(config['train_data'])
    train_loader = DataLoader(
        data, batch_size=config['batchsize'], shuffle=True, num_workers=config['num_workers'])

    # Def. of Model and optimizer
    model = yu.load_component(config['model'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), config['lr'])
    
    manager = ppe.training.ExtensionsManager(
        model, optimizer, None,
        iters_per_epoch=len(train_loader),
        out_dir=config['log_dir'],
        stop_trigger=(config['max_iteration'], 'iteration')
    )

    manager.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'train/loss', 'train/loss_bd', 'train/loss_orth', 'loss_internal_0', 'loss_internal_T', 'elapsed_time']),
        trigger=(config['report_freq'], 'iteration'))
    manager.extend(extensions.LogReport(
        trigger=(config['report_freq'], 'iteration')))
    manager.extend(
        extensions.snapshot(
            target=model, filename='snapshot_model_iter_{.iteration}'),
        trigger=(config['model_snapshot_freq'], 'iteration'))
    manager.extend(
        extensions.snapshot(
            target=manager, filename='snapshot_manager_iter_{.iteration}', n_retains=1),
        trigger=(config['manager_snapshot_freq'], 'iteration'))
    # Run training loop
    print("Start training...")
    yu.load_component_fxn(config['training_loop'])(
        manager, model, optimizer, train_loader, config, device)


if __name__ == '__main__':
    # Loading the configuration arguments from specified config path
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--config_path', type=str)
    parser.add_argument('-a', '--attrs', nargs='*', default=())
    parser.add_argument('-w', '--warning', action='store_true')
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    config['config_path'] = args.config_path
    config['log_dir'] = args.log_dir

    # Modify the yaml file using attr
    for attr in args.attrs:
        module, new_value = attr.split('=')
        keys = module.split('.')
        target = functools.reduce(dict.__getitem__, keys[:-1], config)
        if keys[-1] in target.keys():
            target[keys[-1]] = yaml.safe_load(new_value)
        else:
            raise ValueError('The following key is not defined in the config file:{}', keys)

    for k, v in sorted(config.items()):
        print("\t{} {}".format(k, v))

    # create the result directory and save yaml
    if not os.path.exists(config['log_dir']):
        os.makedirs(config['log_dir'])

    _config = copy.deepcopy(config)
    configpath = os.path.join(config['log_dir'], "config.yml")
    open(configpath, 'w').write(
        yaml.dump(_config, default_flow_style=False)
    )

    # Training
    train(config)
