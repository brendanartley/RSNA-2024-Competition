import os
import pickle
from tqdm import tqdm
import random

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Stage1Dataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode

        if mode == "train":
            self.aug = cfg.train_aug
        else:
            self.aug = cfg.val_aug

        # Mappings
        self.plane2sd= {"s2":"Sagittal T2/STIR", "s1":"Sagittal T1", "a2":"Axial T2"}
        self.sd2plane= {'Sagittal T2/STIR': 's2', 'Sagittal T1': 's1', 'Axial T2': 'a2'}
        self.label2idx= {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
        self.idx2label= {0: 'Normal/Mild', 1: 'Moderate', 2: 'Severe'}

        # Load data
        self.records = self.load_records(df)
        self.labels= self.load_labels()

        print(len(self.records), len(self.labels))

    def load_labels(self,):
        if self.mode in ["train", "val"]:
            # Load coordinate data
            df= pd.read_csv("./data/metadata/coords_v7.csv")

            # Remove noisy series_ids
            df= df[~df.series_id.isin([
                3892989905, 2097107888, 2679683906, 1771893480, 
                996418962, 1753543608, 1848483560, # Bad Axial T2s
                ])]
            df= df[~df.study_id.isin([2492114990, 2780132468, 3008676218, 3637444890])]

            # Filter series_id type
            md= pd.read_csv("./data/raw/train_series_descriptions.csv")
            vals= md.loc[md["series_description"] == self.plane2sd[self.cfg.plane], "series_id"].values
            df= df[df.series_id.isin(vals)]

            def helper(x):
                # Sort by level then relative_x
                arr= sorted(list(x.itertuples(index=False, name=None)), key=lambda x: (x[0], x[1]))
                arr= [_[1:] for _ in arr]
                arr= np.array(arr).astype(float)
                return arr

            # Load coordinates
            d = df.groupby("series_id")[["level", "relative_x", "relative_y"]].apply(lambda x: helper(x))
            for k,v in d.items():
                assert len(v) == self.cfg.n_classes // 2

        else:
            # Create empty labels to inference
            d= {}
            for v in self.records.values():
                d[v["series_id"]]= np.zeros((self.cfg.n_classes//2,2))

        return d

    def load_records(self, df):
        if self.mode in ["train", "val"]:
            # Remove noisy series_ids
            df= df[~df.series_id.isin([
                3892989905, 2097107888, 2679683906, 1771893480, 
                996418962, 1753543608, 1848483560, # Bad Axial T2s
                ])]
            df= df[~df.study_id.isin([2492114990, 2780132468, 3008676218, 3637444890])]

            # Filter series_id type
            md= pd.read_csv("./data/raw/train_series_descriptions.csv")
            vals= md.loc[md["series_description"] == self.plane2sd[self.cfg.plane], "series_id"].values
            df= df[df.series_id.isin(vals)]

            # Load 1 entry per study_id
            records= {}
            for _, row in df.reset_index(drop=True).iterrows():
                records[row.series_id]= {"Sagittal T2/STIR":[], "Sagittal T1":[],"Axial T2":[]}
                records[row.series_id]["series_id"] = row.series_id
                records[row.series_id]["study_id"] = row.study_id
                records[row.series_id][row.series_description].append(row.series_id)
            records= {i:v for i, (k,v) in enumerate(records.items())} # make accessible with idx in __getitem__

        else:
            # Load 1 entry per series_id
            records= {}
            i= 0
            for j in range(self.cfg.n_tta):
                for _, row in df.reset_index(drop=True).iterrows():
                    records[i]= {"Sagittal T2/STIR":[], "Sagittal T1":[],"Axial T2":[]}
                    records[i]["series_id"] = row.series_id
                    records[i]["study_id"] = row.study_id
                    records[i][row.series_description].append(row.series_id)
                    i+=1
        return records

    def load_img(self, study_id, plane, series_id):
        # Load img
        fname= os.path.join(self.cfg.img_dir, "{}/{}.npy".format(study_id, series_id))
        img= np.load(fname)

        # Create mask
        img, mask= self.pad_to_length(img)
        return img, mask

    def get_labels(self, series_id):
        label= self.labels[series_id]
        return label

    def pad_to_length(self, img):
        n = img.shape[-1]
        mask = np.ones(n, dtype=bool)

        if n < self.cfg.n_frames:
            pad_left = (self.cfg.n_frames - n) // 2
            pad_right = self.cfg.n_frames - n - pad_left
            img = np.pad(img, ((0, 0), (0, 0), (pad_left, pad_right)), 'constant', constant_values=0)
            mask = np.pad(mask, (pad_left, pad_right), 'constant', constant_values=False)
        else:
            start = (n - self.cfg.n_frames) // 2
            img = img[..., start:start+self.cfg.n_frames]
            mask = mask[start:start+self.cfg.n_frames]

        return img, mask

    def select_series_id(self, d, plane):
        plane_full= self.plane2sd[plane]
        if self.mode == "train": 
            series_id= random.choice(d[plane_full])
        else: 
            series_id= d[plane_full][0]
        return series_id

    def add_channel_dims(self, img, axis=1):
        img= np.expand_dims(img, axis)
        return img

    def __getitem__(self, idx):
        
        # Load label and metadata
        d= self.records[idx]
        label = self.get_labels(d["series_id"])
        series_id= self.select_series_id(d, plane=self.cfg.plane)

        # Load img
        img, mask= self.load_img(
            study_id= d["study_id"], 
            series_id= series_id, 
            plane= self.cfg.plane,
            )

        # Transform
        if self.aug:

            # Aug img + keypoints
            transformed= self.aug(image=img, keypoints=label*self.cfg.img_size)
            img= transformed["image"].transpose(2,0,1)
            img= img / 255.0
            label= transformed["keypoints"]

            # Normalize between 0-1
            label= (np.array(label) / self.cfg.img_size).flatten()
            label= np.clip(label, 0., 1.)
            
            # Flipping sequence order (does not matter for sagittal)
            # Sag T2 targets: center frame
            # Sag T1 targets: mean of left/right
            if self.mode == "train":
                if random.random() > 0.5:
                    mask= mask[::-1].copy()
                    img= img[::-1, ...].copy()

            # Add channel dim
            img= self.add_channel_dims(img)

        return {
            'input': img, 
            'mask': mask,
            'target': label,
            'series_id': series_id,
            }

    def __len__(self,):
        return len(self.records)


if __name__ == "__main__":
    from src.configs.cfg_stage1_s2 import cfg
    df= pd.read_csv("./data/raw/train_series_descriptions.csv")
    df= df[df.series_description == "Sagittal T2/STIR"]
    ds= Stage1Dataset(df, cfg, mode="train")

    for i in range(min(len(ds), 1)):
        for k, v in ds[i].items():
            try: print(k, v.shape, v.min(), v.max())
            except: print(k, v)
            pass