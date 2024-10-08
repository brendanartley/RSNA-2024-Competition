import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Simple attention.
    """
    def __init__(self, cfg, emb_dim, attn_dropout: float = 0.0):
        super().__init__()
        self.cfg = cfg
        self.attn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Dropout(attn_dropout),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, x, mask: torch.Tensor = None):
        a= self.attn(x)

        # Optional: Masking
        if mask is not None:
            a = a.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        a= torch.softmax(a, dim=1)
        x= torch.sum(x * a, dim=1)
        return x, a

class AttentionLSTM(nn.Module):
    """
    Attention w/ LSTM.
    """
    def __init__(
        self, 
        cfg, 
        emb_dim, 
        attn_dropout: float = 0.0, 
        rnn_dropout: float = 0.0, 
        bidirectional: float = True,
        ):
        super().__init__()
        self.cfg = cfg
        self.attn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Tanh(),
            nn.Dropout(attn_dropout),
            nn.Linear(emb_dim, 1)
        )

        if bidirectional:
            hidden_dim= emb_dim//2
        else:
            hidden_dim= emb_dim

        self.rnn = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=bidirectional, dropout=0.0)
        self.rnn_dropout = nn.Dropout(rnn_dropout)
        self.layer_norm = nn.LayerNorm(emb_dim)

    def forward(self, x, mask: torch.Tensor = None):
        # Rnn
        x, _ = self.rnn(x)
        x = self.rnn_dropout(x)
        x = F.mish(x) + F.sigmoid(x) + F.tanh(x)
        x = self.layer_norm(x)

        # Attn
        a= self.attn(x)

        # Optional: Masking
        if mask is not None:
            a = a.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        a= torch.softmax(a, dim=1)
        x= torch.sum(x * a, dim=1)
        return x, a

if __name__ == "__main__":
    from src.configs.cfg_stage2_s1_fo import cfg

    cfg.rnn_cat= False

    emb_dim= 1024
    att_dim= 512
    seq_len= 24
    x= torch.ones(2,seq_len,emb_dim)
    mask= torch.ones(seq_len, dtype=bool)
    mask[-2:]= False

    l1= Attention(cfg=cfg, emb_dim=emb_dim)
    z, a = l1(x, mask=mask)

    print(z.shape, a.shape)
