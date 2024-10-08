import os
import pickle
from tqdm import tqdm
import random
import glob

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import albumentations as A
import cv2

class Stage2Dataset(torch.utils.data.Dataset):
    def __init__(self, df, cfg, mode="train"):
        self.cfg = cfg
        self.mode = mode

        if mode == "train":
            self.aug = cfg.train_aug
        else:
            self.aug = cfg.val_aug

        # Mappings
        self.plane2sd= {"s2":"Sagittal T2/STIR", "s1":"Sagittal T1", "a2":"Axial T2"}
        self.sd2plane= {'Sagittal T2/STIR':'s2', 'Sagittal T1':'s1', "Axial T2":"a2"}
        self.label2idx= {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
        self.idx2label= {0: 'Normal/Mild', 1: 'Moderate', 2: 'Severe'}

        # Label cols
        self.label_cols= [
            'spinal_canal_stenosis_l1_l2', 'left_neural_foraminal_narrowing_l1_l2', 'right_neural_foraminal_narrowing_l1_l2', 
            'left_subarticular_stenosis_l1_l2', 'right_subarticular_stenosis_l1_l2', 'spinal_canal_stenosis_l2_l3', 
            'left_neural_foraminal_narrowing_l2_l3', 'right_neural_foraminal_narrowing_l2_l3', 'left_subarticular_stenosis_l2_l3',
            'right_subarticular_stenosis_l2_l3',
            'spinal_canal_stenosis_l3_l4', 'left_neural_foraminal_narrowing_l3_l4', 'right_neural_foraminal_narrowing_l3_l4',
            'left_subarticular_stenosis_l3_l4', 'right_subarticular_stenosis_l3_l4', 'spinal_canal_stenosis_l4_l5',
            'left_neural_foraminal_narrowing_l4_l5', 'right_neural_foraminal_narrowing_l4_l5', 'left_subarticular_stenosis_l4_l5', 
            'right_subarticular_stenosis_l4_l5', 'spinal_canal_stenosis_l5_s1', 'left_neural_foraminal_narrowing_l5_s1',
            'right_neural_foraminal_narrowing_l5_s1', 'left_subarticular_stenosis_l5_s1', 'right_subarticular_stenosis_l5_s1',
        ]

        if self.cfg.condition == "all":
            self.keep_idxs= np.array([True]*25)
        else:
            self.keep_idxs= np.array([True if self.cfg.condition in x else False for x in self.label_cols])
            self.label_cols= [x for x in self.label_cols if self.cfg.condition in x]
            
        self.left_idxs= [i for i,val in enumerate(self.label_cols) if "left" in val]
        self.right_idxs= [i for i,val in enumerate(self.label_cols) if "right" in val]

        # Load data
        self.records, self.labels = self.load_records(df)

    def load_records(self, df):
        if self.mode in ["train", "val"]:
            # Record dict
            df= df[~df.series_id.isin([
                3892989905, 2097107888, 2679683906, 1771893480, 
                996418962, 1753543608, 1848483560, # Bad Axial T2s
                ])]
            df= df[~df.study_id.isin([2492114990, 2780132468, 3008676218, 3637444890])]

        # Load records
        if self.mode == "train":
            records = {x:{"Sagittal T2/STIR":[], "Sagittal T1":[],"Axial T2":[]} for x in df["study_id"].unique()}
            for i, row in df.iterrows():
                records[row.study_id]["study_id"] = row.study_id
                records[row.study_id][row.series_description].append(row.series_id)
                records[row.study_id]["tta"]= 0
            records= {i:v for i, (k,v) in enumerate(records.items())} # make accessible with idx in __getitem__

        # Predict on every series_id
        else:
            records= {}
            i= 0
            for j in range(self.cfg.n_tta):
                for _, row in df.reset_index(drop=True).iterrows():
                    records[i]= {"Sagittal T2/STIR":[], "Sagittal T1":[],"Axial T2":[]}
                    records[i]["study_id"] = row.study_id
                    records[i][row.series_description].append(row.series_id)
                    records[i]["tta"]= j
                    i+=1

        # Load Labels
        if self.mode== "test":
            labels= {}
            for v in records.values():
                labels[v["study_id"]]= np.zeros((self.cfg.n_classes//3)).astype(int)
        else:
            label_df= pd.read_csv("./data/raw/train.csv")

            # Remove noisy labels
            drop_df= pd.read_csv("./data/metadata/noisy_target_level_threshold0.85.csv")
            for row in drop_df.itertuples():
                col= row.target + "_" + row.level
                label_df.loc[label_df.study_id == row.study_id, col]= ""
            
            # Fill NANs with -100
            for col in self.label_cols:
                label_df[col]= label_df[col].apply(lambda x: self.label2idx.get(x, -100)).astype(int)
            labels= label_df[self.label_cols].values
            print("NANs Labels:", np.sum(labels == -100))
            # Create label dict
            labels= {x:y for x,y in zip(label_df["study_id"].values, labels)}

        return records, labels

    def load_img(self, study_id, plane, series_id):
        fname= os.path.join(self.cfg.img_dir, "{}/{}.npy".format(study_id, series_id))
        img= np.load(fname)
        img, mask= self.pad_to_length(img)

        return img, mask

    def pad_to_length(self, img):
        n = img.shape[-1]
        mask = np.ones(n, dtype=bool)

        if n < self.cfg.n_frames:
            pad_left = (self.cfg.n_frames - n) // 2
            pad_right = self.cfg.n_frames - n - pad_left
            img = np.pad(img, ((0, 0), (0, 0), (0, 0), (pad_left, pad_right)), 'constant', constant_values=0)
            mask = np.pad(mask, (pad_left, pad_right), 'constant', constant_values=False)
        else:
            start = (n - self.cfg.n_frames) // 2
            img = img[..., start:start+self.cfg.n_frames]
            mask = mask[:self.cfg.n_frames]

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

    def get_labels(self, study_id, series_id):
        # Label
        label= self.labels[study_id]

        # OHE
        ohe_label= np.eye(3)[label.clip(min=0)].astype(float)
        label_mask = (label == -100) # Mask out missing label

        # Mask
        ohe_label[label_mask] = -100

        return ohe_label

    def reverse_labels(self, label):
        label2= label.copy()
        label2[self.left_idxs]= label[self.right_idxs]
        label2[self.right_idxs]= label[self.left_idxs]
        label= label2
        return label

    def _apply_tta(self, img, tta: int):
        # Rotation Angles
        angles= [ 0., -15., 15., -10.,  10., -5.,  5.,  -20, 20.]

        if tta > 8:
            raise ValueError("Max reproducible TTA is 9.")

        transform = A.Compose([
            A.Rotate(limit=(angles[tta], angles[tta]), always_apply=True),
            A.Resize(self.cfg.img_size, self.cfg.img_size, p=1.0, interpolation=cv2.INTER_CUBIC),
        ])

        # Transform
        h,w,lvl,seq= img.shape
        img= img.reshape(h,w,lvl*seq)
        t= transform(image=img)
        img= t["image"].reshape(self.cfg.img_size,self.cfg.img_size,lvl,seq)

        return img

    def __getitem__(self, idx):
        # Load label and metadata
        d= self.records[idx]
        series_id= self.select_series_id(d, plane=self.cfg.plane)
        label = self.get_labels(d["study_id"], series_id)

        # Load img
        img, mask= self.load_img(
            study_id= d["study_id"], 
            series_id= series_id, 
            plane= self.cfg.plane,
            )

        # Manual Augs
        if self.mode == "train":

            # Sagittal: flip sequence order
            if random.random() < 0.5:
                img= img[:, :, :, ::-1].copy()
                mask= mask[::-1].copy()
                label= self.reverse_labels(label)

        # TTA
        else:
            img= self._apply_tta(img, tta= d["tta"])

        # Albumentations augs
        if self.aug:
            h,w,lvl,seq= img.shape
            img= img.reshape(h,w,lvl*seq)
            t= self.aug(image=img)
            img= t["image"].transpose(2,0,1) / 255.0
            img= img.reshape(lvl,seq,self.cfg.img_size,self.cfg.img_size)

        # Unsqueeze channel dim
        img= self.add_channel_dims(img, axis=2)
        
        # Reshape so disc level is first dimension
        label= label.reshape(5, -1)

        return {
            'input': img, 
            'target': label,
            'mask': mask,
            'series_id': series_id,
            'study_id': int(d["study_id"]),
            'tta': d["tta"],
            }
    
    def __len__(self,):
        return len(self.records)


if __name__ == "__main__":
    from src.configs.cfg_stage2_s2_sp1 import cfg
    df= pd.read_csv("./data/raw/train_series_descriptions.csv")
    df= df[df.series_description == "Sagittal T2/STIR"]
    df= df.head(1)

    for mode in ["train"]:
        ds= Stage2Dataset(df, cfg, mode=mode)
        # print(ds.labels)
        for i in range(min(len(ds), 10)):
            for k, v in ds[i].items():
                # try: print(k, v.shape, v.min(), v.max())
                # except: print(k, v)
                pass