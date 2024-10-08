from typing import Iterable
import json
import glob
from importlib import import_module

import torch
import pandas as pd

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model(cfg, pretrained: bool = True, inference_mode: bool = False):

    # Build
    m = import_module(f"src.models.{cfg.model_type}").Net(
        cfg=cfg, 
        pretrained=pretrained, 
        inference_mode=inference_mode,
        )

    # Param count
    n_params= count_parameters(m)
    print(f"Model: {cfg.model_type}")
    print("n_param: {:_}".format(n_params))

    # Load weights
    f= cfg.weights_path
    if f != "":
        m.load_state_dict(torch.load(f, map_location=cfg.device))
        print("LOADED WEIGHTS:", f)

    return m, n_params

def batch_to_device(batch, device, skip_keys: Iterable[str]= []):
    batch_dict= {}
    for key in batch:
        if key in skip_keys:
             batch_dict[key]= batch[key]
        else:    
            batch_dict[key]= batch[key].to(device)
    return batch_dict

def calc_grad_norm(parameters,norm_type=2.):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    if torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        total_norm = None
        
    return total_norm

def get_optimizer(model, cfg):
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    return optimizer

def get_scheduler(optimizer, cfg, n_steps):
    if cfg.scheduler == "Constant":
        s= torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    elif cfg.scheduler == "CosineAnnealingLR":
        s= torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max = n_steps,
            eta_min = cfg.lr_min,
            )
    else:
        raise ValueError(f"{cfg.scheduler} is not a valid scheduler.")
    return s

def flatten_dict(d):
    def _flatten(current_key, nested_dict, flattened_dict):
        for k, v in nested_dict.items():
            new_key = f"{current_key}.{k}" if current_key else k
            if isinstance(v, dict) and v:
                _flatten(new_key, v, flattened_dict)
            elif v is not None and v != {}:  # Exclude None values and empty dictionaries
                flattened_dict[new_key] = v
    
    flattened_dict = {}
    _flatten("", d, flattened_dict)
    return flattened_dict


def save_weights(
    model, 
    cfg,
    train_metrics: dict = {},
    val_metrics: dict = {},
    ):

    # Create fpaths
    fpath = f"./data/checkpoints/{cfg.config_file}_fold{cfg.fold}_seed{cfg.seed}.pt"
    meta_fpath = fpath.replace(".pt", ".json")

    # Save weights
    torch.save(model.state_dict(), fpath)
    print(f"SAVED WEIGHTS: {fpath}")

    # Save metadata
    metadata = {
        'config_file': cfg.config_file,
        'model_type': cfg.model_type,
        'attn_type': cfg.attn_type,
        'backbone': cfg.backbone,
        'fold': cfg.fold,
        'mixup_prob': cfg.mixup_prob,
        'cutmix_prob': cfg.cutmix_prob,
        'seed': cfg.seed,
        'fold': cfg.fold,
        'metrics': {
            "train": train_metrics,
            "val": val_metrics,
        }
    }
    with open(meta_fpath, 'w') as f:
        json.dump(metadata, f, indent=4)

    return fpath

if __name__ == "__main__":
    pass