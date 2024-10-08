import argparse
from copy import copy
import sys
import importlib
import os
import random
import numpy as np
import torch
import timm

torch.set_printoptions(sci_mode=False)

def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-C", "--config", help="config filename", default="cfg_stage2_s2_sp1")
    parser.add_argument("-G", "--gpu_id", default="", help="GPU ID")
    parser_args, other_args = parser.parse_known_args(sys.argv)

    # Use all GPUs unless specified
    if parser_args.gpu_id != "":
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parser_args.gpu_id)

    # Load CFG
    cfg = copy(importlib.import_module('src.configs.{}'.format(parser_args.config)).cfg)
    cfg.config_file = parser_args.config
    print("config ->", cfg.config_file)

    # Overwrite other arguments
    if len(other_args) > 1:
        other_args = {v.split("=")[0].lstrip("-"):v.split("=")[1] for v in other_args[1:]}

        for key in other_args:
            
            # Nested config
            if "." in key:
                keys = key.split(".")
                assert len(keys) == 2

                print(f'overwriting cfg.{keys[0]}.{keys[1]}: {cfg.__dict__[keys[0]].__dict__[keys[1]]} -> {other_args[key]}')
                cfg_type = type(cfg.__dict__[keys[0]].__dict__[keys[1]])
                if cfg_type == bool:
                    cfg.__dict__[keys[0]],__dict__[keys[1]] = other_args[key] == 'True'
                elif cfg_type == type(None):
                    cfg.__dict__[keys[0]].__dict__[keys[1]] = other_args[key]
                else:
                    cfg.__dict__[keys[0]].__dict__[keys[1]] = cfg_type(other_args[key])
                print(cfg.__dict__[keys[0]].__dict__[keys[1]])

            # Main config
            elif key in cfg.__dict__:
                print(f'overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}')
                cfg_type = type(cfg.__dict__[key])
                if cfg_type == bool:
                    cfg.__dict__[key] = other_args[key] == 'True'
                elif cfg_type == type(None):
                    cfg.__dict__[key] = other_args[key]
                else:
                    cfg.__dict__[key] = cfg_type(other_args[key])
                print(cfg.__dict__[key])
    
    # Set seed
    if cfg.seed < 0:
        cfg.seed = np.random.randint(1_000_000)
    print("seed", cfg.seed)
    set_seed(cfg.seed)

    # Quick development run
    if cfg.fast_dev_run:
        cfg.epochs= 1
        cfg.no_wandb= None

    return cfg

if __name__ == "__main__":
    cfg= parse_args()
    
    if cfg.project == "rsna":
        from src.modules.train_stage2 import train
        train(cfg)

    elif cfg.project == "rsna_localizer":
        from src.modules.train_stage1 import train
        train(cfg)