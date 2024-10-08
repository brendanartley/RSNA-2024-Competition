import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, cfg, ignore_index= -100):
        super().__init__()
        self.cfg= cfg
        self.weights = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float32).to(cfg.device)
        self.ignore_index= ignore_index

    def _calc_sample_weights(self, targets):
        # Argmax for sample weights
        # Note: Hacky way to use sample weights with cutmix/mixup/noisy_student
        mask= targets != 0.0
        masked_targets= torch.where(mask, targets, torch.tensor(float('-inf')))
        ohe_targets= torch.argmax(masked_targets, dim=1)

        # Mask out -100s
        mask= targets[:, 0] != -100
        weights= torch.zeros_like(ohe_targets, dtype=torch.float32)
        weights[mask]= self.weights[ohe_targets[mask]]

        return weights

    def forward(self, inputs, targets):
        # Calc sample weights
        sample_weights = self._calc_sample_weights(targets)

        # CE loss: Targets
        log_probs= F.log_softmax(inputs, dim=-1)
        ce_loss= -1 * torch.sum(targets * log_probs, dim=-1) # CE with targets
        loss= ce_loss

        # Sample weights + mask
        loss= loss*sample_weights
        
        return loss.mean()


if __name__ == "__main__":
    from src.configs.cfg_stage2_s2_sp1 import cfg
    n= 25
    logits= torch.randn(n*3).reshape(n,3).to(cfg.device)
    y = torch.randint(low=0, high=2, size=(n,)).to(cfg.device)
    y_ohe = F.one_hot(y, num_classes=3).float().to(cfg.device)
    print(logits.shape, y.shape)

    # Ignoring a few indexes
    y[0, [5,7,9]] = -100.0

    loss_fn= WeightedCrossEntropyLoss(cfg)
    l = loss_fn(logits, y_ohe)

    print(l)
