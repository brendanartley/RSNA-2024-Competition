import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace

from src.models.utils import (
    get_backbone, 
)

from importlib import import_module
attn_modules = import_module("src.models.attn")

class Net(nn.Module):
    def __init__(
        self, 
        cfg: SimpleNamespace, 
        pretrained: bool = True, 
        inference_mode: bool = False,
        ):
        super().__init__()
        self.cfg= cfg
        self.loss_fn= self._init_loss_fn()

        self.backbone, emb_dim= get_backbone(cfg, pretrained=pretrained)
        self.attn = getattr(attn_modules, self.cfg.attn_type)(
            cfg=cfg, 
            emb_dim=emb_dim,
            )
        
        self.fc= nn.Linear(emb_dim, cfg.n_classes)
        self.fc_dropout= nn.Dropout(p= cfg.fc_dropout)

        self.inference_mode= inference_mode

    def _init_loss_fn(self):
        return nn.L1Loss()

    def forward(self, batch):
        x= batch["input"].float()
        y= batch["target"].float()
        mask= batch["mask"]
        b,seq,c,h,w= x.shape

        # Backbone
        x = x.view(b*seq, c, h, w)
        x = self.backbone(x)
        x = self.fc_dropout(x)

        # Attn
        x = x.view(b, seq, -1)
        x, _= self.attn(x, mask= mask)

        # FC
        x = self.fc(x)
        x = torch.sigmoid(x) # NOTE: Needed for relative x,y pred

        # Return dict
        if self.inference_mode:
            loss= 0
        else:
            loss= self.loss_fn(x, y)
        outputs= {
            "loss": loss,
            "logits": x,
        }

        return outputs