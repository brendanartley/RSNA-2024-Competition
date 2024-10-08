import torch
import numpy as np
import pandas as pd

from src.data.dataset_stage1 import Stage1Dataset
from src.data.dataset_stage2 import Stage2Dataset

def get_data(cfg):
    # Load metadata
    df = pd.read_csv(cfg.train_df)
    
    pmap= {"s2":"Sagittal T2/STIR", "s1":"Sagittal T1", "a2":"Axial T2"}
    planes= [pmap[x] for x in cfg.plane.strip().split(",")]
    df= df[df["series_description"].isin(planes)]
    
    # Overfitting a single batch
    if cfg.overfit_batch:
        return df.head(cfg.batch_size), df.head(cfg.batch_size)

    # Train
    train_df = df[df["fold"] != cfg.fold]

    # Val
    if cfg.fold == -1:
        val_df = df[df["fold"] == 0]
    else:
        val_df = df[df["fold"] == cfg.fold]

    if cfg.fast_dev_run:
        train_df= train_df.head(cfg.batch_size)
        val_df= val_df.head(cfg.batch_size)
        
    return train_df, val_df

def get_dataset(df, cfg, mode='train'):
    if mode == 'train':
        dataset = get_train_dataset(df, cfg)
    elif mode == 'val':
        dataset = get_val_dataset(df, cfg)
    else:
        pass
    return dataset

def get_dataloader(ds, cfg, mode='train'):
    if mode == 'train':
        dl = get_train_dataloader(ds, cfg)
    elif mode =='val':
        dl = get_val_dataloader(ds, cfg)
    return dl

def get_train_dataset(train_df, cfg):
    if cfg.project == "rsna":
        train_dataset = Stage2Dataset(train_df, cfg, mode="train")
    elif cfg.project == "rsna_localizer":
        train_dataset = Stage1Dataset(train_df, cfg, mode="train")

    if cfg.data_sample > 0:
        train_dataset = torch.utils.data.Subset(train_dataset, np.arange(cfg.data_sample))
    return train_dataset

def get_train_dataloader(train_ds, cfg):
    # TODO: Add weighted sampler option
    train_dataloader = torch.utils.data.DataLoader(
        train_ds,
        sampler= None,
        shuffle= True,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
    )
    print(f"TRAIN: dataset {len(train_ds)} dataloader {len(train_dataloader)}")
    return train_dataloader

def get_val_dataset(val_df, cfg):
    if cfg.project == "rsna":
        val_dataset = Stage2Dataset(val_df, cfg, mode="val")
    elif cfg.project == "rsna_localizer":
        val_dataset = Stage1Dataset(val_df, cfg, mode="val")
        
    return val_dataset

def get_val_dataloader(val_ds, cfg):
    sampler = torch.utils.data.SequentialSampler(val_ds)

    if cfg.batch_size_val is not None:
        batch_size = cfg.batch_size_val
    else:
        batch_size = cfg.batch_size
    val_dataloader = torch.utils.data.DataLoader(
        val_ds,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    print(f"VALID: dataset {len(val_ds)}, dataloader {len(val_dataloader)}")
    return val_dataloader


if __name__ == "__main__":
    from src.configs.cfg_default import cfg

    df,val_df = get_data(cfg)