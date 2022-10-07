import os
import sys
import functools
import argparse
import yaml
import pdb
sys.path.append('../')
sys.path.append('./')

# Originally created by @msaito
def load_module(fn, name):
    mod_name = os.path.splitext(os.path.basename(fn))[0]
    mod_path = os.path.dirname(fn)
    sys.path.insert(0, mod_path)
    return getattr(__import__(mod_name), name)


def load_component(config):
    class_fn = load_module(config['fn'], config['name'])
    return class_fn(**config['args']) if 'args' in config.keys() else class_fn()


def load_component_fxn(config):
    fxn = load_module(config['fn'], config['name'])
    return fxn


def make_function(module, name):
    fxn = getattr(module, name)
    return fxn


def make_instance(module, config=[], args=None):
    Class = getattr(module, config['name'])
    kwargs = config['args']
    if args is not None:
        kwargs.update(args)
    return Class(**kwargs)


'''
conbines multiple configs
'''


def make_config(conf_dicts, attr_lists=None):
    def merge_dictionary(base, diff):
        for key, value in diff.items():
            if (key in base and isinstance(base[key], dict)
                    and isinstance(diff[key], dict)):
                merge_dictionary(base[key], diff[key])
            else:
                base[key] = diff[key]

    config = {}
    for diff in conf_dicts:
        merge_dictionary(config, diff)
    if attr_lists is not None:
        for attr in attr_lists:
            module, new_value = attr.split('=')
            keys = module.split('.')
            target = functools.reduce(dict.__getitem__, keys[:-1], config)
            target[keys[-1]] = yaml.load(new_value)
    return config


'''
argument parser that uses make_config
'''


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'infiles', nargs='+', type=argparse.FileType('r'), default=())
    parser.add_argument('-a', '--attrs', nargs='*', default=())
    parser.add_argument('-c', '--comment', default='')
    parser.add_argument('-w', '--warning', action='store_true')
    parser.add_argument('-o', '--output-config', default='')
    args = parser.parse_args()

    conf_dicts = [yaml.load(fp) for fp in args.infiles]
    config = make_config(conf_dicts, args.attrs)
    return config, args
