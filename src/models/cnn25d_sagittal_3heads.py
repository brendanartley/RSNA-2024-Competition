import torch
import torch.nn as nn
import torch.nn.functional as F

from types import SimpleNamespace
from typing import Iterable
import numpy as np

from src.models.utils import (
    get_backbone, 
)

from src.models.loss import (
    WeightedCrossEntropyLoss,
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
        assert self.cfg.condition == "all"

        # Spinal
        self.attn1 = getattr(attn_modules, self.cfg.attn_type)(
            cfg=cfg, 
            emb_dim=emb_dim,
            )
        self.fc1= nn.Linear(emb_dim, 3)
        
        # Foraminal
        self.attn2 = getattr(attn_modules, self.cfg.attn_type)(
            cfg=cfg, 
            emb_dim=emb_dim,
            )
        self.fc2= nn.Linear(emb_dim, 6)

        # Subarticular
        self.attn3 = getattr(attn_modules, self.cfg.attn_type)(
            cfg=cfg, 
            emb_dim=emb_dim,
            )
        self.fc3= nn.Linear(emb_dim, 6)
        self.fc_dropout = nn.Dropout(p= cfg.fc_dropout)

        self.inference_mode = inference_mode

    def _init_loss_fn(self):
        return WeightedCrossEntropyLoss(
            cfg= self.cfg,
            ignore_index= self.cfg.ignore_index,
            )

    def mixup(self, batch: Iterable[torch.Tensor], target: torch.Tensor, proba: float, alpha: float):
        gamma = np.random.beta(alpha, alpha)

        # Mask missing values
        mask = torch.min(target, dim=-1).values != -100
        mask = mask & mask.roll(1, 0)
        mask &= torch.rand_like(mask, dtype=torch.float) < proba

        # .roll() is faster than .randperm()
        batch_rolled = [_.roll(1, 0) for _ in batch]
        target_rolled = target.roll(1, 0)

        # Apply mixup with mask condition
        batch = [
            torch.where(mask[:, None], b.mul(gamma).add(br, alpha=1-gamma), b)
            for b,br in zip(batch, batch_rolled)
        ]
        target = torch.where(mask[:, None], target.mul(gamma).add(target_rolled, alpha=1-gamma), target)

        return batch, target

    def cutmix(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, proba: float):

        # Create mask
        cut_mask= torch.rand(y.shape[0], dtype=torch.float) < proba
        cut_mask= cut_mask.to(x.device)
        idxs = torch.randperm(x.shape[0], device=x.device) # shuffles between levels + series
        x_roll= x[idxs]
        y_roll= y[idxs]
        mask_roll= mask[idxs]

        # Split on midpoint
        mid= x.shape[1]//2
        x_mixed= torch.cat([x[:, :mid], x_roll[:, mid:]], dim=1)
        mid= y.shape[-1]//2
        y_mixed= torch.cat([y[:, :6], y_roll[:, 6:9], y[:, 9:12], y_roll[:, 12:15]], dim=1)
        mid = mask.shape[1]//2
        mask_mixed = torch.cat([mask[:, :mid], mask_roll[:, mid:]], dim=1)

        # Combine 
        x= torch.where(cut_mask[:, None, None], x_mixed, x)
        y= torch.where(cut_mask[:, None], y_mixed, y)
        mask= torch.where(cut_mask[:, None], mask_mixed, mask)

        return x, y, mask

    def forward(self, batch):
        x= batch["input"].float()
        y= batch["target"]
        mask= torch.repeat_interleave(batch["mask"], repeats=5, dim=0) # repeat mask for 5 levels
        b,vert,seq,c,h,w= x.shape

        # Backbone
        x = x.view(b*vert*seq, c, h, w)
        x = self.backbone(x)
        x = self.fc_dropout(x)
        x = x.view(b*vert, seq, -1)

        # Spinal (Cutmix doesnt make sense for spinal)
        x1, a1= self.attn1(x, mask= mask) 

        # Cutmix
        if self.training:
            y= y.view(b*vert, -1)
            x, y, mask = self.cutmix(x, y, mask, proba= self.cfg.cutmix_prob)

        # Attentions: (Left then right side)
        x2, a2 = self.attn2(x, mask= mask)
        x3, a3 = self.attn3(x, mask= mask)

        # Manifold Mixup
        if self.training:
            y= y.view(x.shape[0], -1)
            _, y = self.mixup(
                [x1, x2, x3],
                y, 
                proba= self.cfg.mixup_prob,
                alpha= self.cfg.mixup_alpha,
                )
            x1, x2, x3= _
            del _

        # FCs
        x1= self.fc_dropout(x1)
        x1= self.fc1(x1)
        x2= self.fc_dropout(x2)
        x2= self.fc2(x2)
        x3= self.fc_dropout(x3)
        x3= self.fc3(x3)

        # Stack
        x= torch.cat([x1, x2, x3], dim=1)

        # Loss
        x= x.reshape(-1,3)
        y= y.reshape(-1,3)

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
