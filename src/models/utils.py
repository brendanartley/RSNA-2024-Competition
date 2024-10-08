from copy import deepcopy

import torch
import torch.nn as nn
import timm

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_backbone(cfg, pretrained: bool = True):
    m= timm.create_model(
        cfg.backbone,
        pretrained= pretrained,
        in_chans= cfg.in_chans,
        drop_path_rate= cfg.drop_path_rate,
    )
    emb_dim= m.head.fc.in_features
    m.head.fc= nn.Identity()

    if cfg.grad_checkpointing:
        m.set_grad_checkpointing()
        
    return m, emb_dim