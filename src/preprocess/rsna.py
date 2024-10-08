import os
os.environ['NO_ALBUMENTATIONS_UPDATE']= "1"

import math
import glob
import json
import pickle
import pandas as pd
import numpy as np
import cv2
import albumentations as A
import pydicom
from tqdm import tqdm
import multiprocessing as mp
from scipy.ndimage import zoom

from typing import Optional, Iterable

from PIL import Image

class RsnaProcessor():
    def __init__(
        self, 
        df: pd.DataFrame,
        coords_sag: pd.DataFrame = None,
        stage: int = 1,
        percentiles: tuple[int]= (1,99),
        mode: str = "train",
        in_dir: str = "./data/raw/", 
        out_dir: str = "./data/processed/", 
        t2_width_pct: float = 75,
        t2_offset_pct: float = 50,
        t2_height_mult: float = 130,
        t1_width_mult: float = 125,
        t1_height_mult: float = 100,
    ):
        super().__init__()
        self.df= df
        self.coords_sag= coords_sag
        self.stage= stage
        self.percentiles= percentiles
        self.mode= mode
        self.in_dir= os.path.join(in_dir, f"{self.mode}_images")
        self.out_dir= out_dir

        # Crop scale/offsets
        self.t2_width_pct= t2_width_pct / 100
        self.t2_offset_pct= t2_offset_pct / 100
        self.t2_height_mult= t2_height_mult / 100

        self.t1_width_mult= t1_width_mult / 100
        self.t1_height_mult= t1_height_mult / 100

        # Stage2: Resize Transform
        self.img_size= 256
        self.resize= A.Compose([
            A.LongestMaxSize(max_size=self.img_size, interpolation=cv2.INTER_LANCZOS4, always_apply=True),
            A.PadIfNeeded(min_height=self.img_size, min_width=self.img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), always_apply=True),
        ])

        # Mode
        modes= ["train", "test"]
        if mode not in modes:
            raise ValueError(f"Mode must be in {str(modes)}")

        # Create main out_dir
        if not os.path.exists(out_dir): 
            os.mkdir(out_dir)

        # Create study_id out_dirs
        for study_id in self.df["study_id"].unique():
            out_dir = os.path.join(self.out_dir, str(study_id))
            if not os.path.exists(out_dir):
                os.mkdir(out_dir)

    def PIL_resize(self, img, img_size, preserve_aspect: bool = True):
        image = Image.fromarray(img)

        h, w = img_size
        if preserve_aspect:
            ow, oh = image.size
            aspect_ratio = ow / oh
            if ow > oh:
                nw = w
                nh = int(w / aspect_ratio)
            else:
                nh = h
                nw = int(h * aspect_ratio)
        else:
            nw, nh = w, h

        resized_image = image.resize((nw, nh), Image.LANCZOS)
        resized_array = np.array(resized_image)
        return resized_array

    def load_dicom_stack(self, dicom_folder, plane, reverse_sort=False):
        # Source: https://www.kaggle.com/code/vaillant/cross-reference-images-in-different-mri-planes
        # Load Dicoms
        dicom_files = glob.glob(os.path.join(dicom_folder, "*.dcm"))
        dicoms = [pydicom.dcmread(f) for f in dicom_files] 
        
        # Get positions
        plane = {"Sagittal T2/STIR": 0, "Sagittal T1": 0, "Axial T2": 2}[plane]

        # Sort by Axials=InstanceNumber, Saggital=IPP
        if plane == 2:
            positions= np.asarray([float(d.InstanceNumber) for d in dicoms])
        else:
            positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
        
        # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
        # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
        idx = np.argsort(-positions)
        dicoms= [dicoms[x] for x in idx]
        ipps= np.asarray([d.ImagePositionPatient[plane] for d in dicoms]).astype("float")

        # IMG stats
        dicom_array=[d.pixel_array.astype("float32") for d in dicoms]
        dicom_array_flat= np.concatenate([x.flatten() for x in dicom_array])
        percentiles = (np.percentile(dicom_array_flat, self.percentiles[0]), np.percentile(dicom_array_flat, self.percentiles[1]))
        
        instance_numbers= [d.InstanceNumber for d in dicoms]

        return {
            "dicoms": dicoms,
            "array": dicom_array,
            "percentiles": percentiles,
            "instance_numbers": instance_numbers,
        }

    def percentile_norm(self, x, percentiles):
        lower, upper= percentiles
        x = np.clip(x, lower, upper)
        x = (x - lower) / (upper - lower)
        return (x*255).astype(np.uint8)

    def angle_of_line(self, x1, y1, x2, y2):
        return math.degrees(math.atan2(-(y2-y1), x2-x1))

    def crop_around_keypoint(self, img, keypoint, y_crop_pct, x_crop_pct, series_description):
        h, w = img.shape[:2]
        x, y = int(keypoint[0]), int(keypoint[1])

        # Show more of the verts/discs
        if series_description == "Sagittal T2/STIR":
            x -= (w*x_crop_pct) * self.t2_offset_pct
        elif series_description == "Axial T2":
            y += (h*y_crop_pct) * 0.35
        
        # Calculate bounding box around the keypoints
        left= int(x - (w * x_crop_pct))
        right = int(x + (w * x_crop_pct))
        top = int(y - (h * y_crop_pct))
        bottom = int(y + (h * y_crop_pct))

        # Keep in bounds
        left= max(0, left)
        right= min(right, w)
        top= max(0, top)
        bottom= min(bottom, h)
                
        # Crop the image
        return img[top:bottom, left:right, ...]

    def middle_n_frames(self, img, n_frames=16):
        n= img.shape[-1]
        if n >= n_frames:
            i = (n - n_frames) // 2
            return img[..., i:i + n_frames]
        else:
            return img

    def load_dcm(self, study_id, series_id, series_description):
        cur_dir= os.path.join(self.in_dir, str(study_id), str(series_id))
        
        d= self.load_dicom_stack(
            dicom_folder= cur_dir, 
            plane= series_description,
            reverse_sort= False,
            )
        dicoms= d["dicoms"]
        images= d["array"]
        percentiles= d["percentiles"]

        # Stage1: Save entire sequence
        if self.stage == 1:

            # Resize/Norm
            images= [
                self.PIL_resize(
                    x, 
                    img_size= (self.img_size, self.img_size), 
                    preserve_aspect= False,
                ) 
                for x in images
                ]
            img= np.stack(images, axis=-1)
            img= self.percentile_norm(img, percentiles)
            img= img.astype(np.uint8)

        # Stage 2:
        # Crop around discs
        elif self.stage == 2:
            img = self.create_5_saggital_crops(
                images, dicoms, series_id, series_description, percentiles,
                )

        return img

    def create_5_saggital_crops(self, img, dicoms, series_id, series_description, percentiles):
        img= np.stack(img, axis=-1)

        # ------------------------- Process Saggital ---------------------------
        cdf= self.coords_sag[self.coords_sag["series_id"] == series_id].copy()

        # Crop PCTs
        if series_description == "Sagittal T2/STIR":
            y_crop_pct= cdf[cdf.side == "R"].groupby(['side'])['relative_y'].diff().median() * self.t2_height_mult
            x_crop_pct= cdf.groupby(['level'])['relative_x'].diff()[1::2].median() * self.t2_width_pct
        elif series_description == "Sagittal T1":
            y_crop_pct= cdf.groupby(['side'])['relative_y'].diff()[::2].median() * self.t1_height_mult
            x_crop_pct= y_crop_pct * self.t1_width_mult
        else:
            raise ValueError("series_description not recognized.")

        # Coords -> Pairs
        p= cdf.groupby("level") \
                      .apply(lambda g: list(zip(g['relative_x'], g['relative_y'])), include_groups=False) \
                      .reset_index(drop=False, name="vals")

        imgs= []
        for idx, (_, row) in enumerate(p.iterrows()):
            # Copy of img
            img_copy= img.copy()
            h, w, n_chans = img.shape

            # Extract Keypoints
            level = row['level']
            vals = sorted(row["vals"], key=lambda x: x[0])
        
            a,b= vals
            a= (a[0]*w, a[1]*h)
            b= (b[0]*w, b[1]*h)
            if a[0] > b[0]:
                a,b=b,a

            # S2 Crop
            if series_description == "Sagittal T2/STIR":
                # Calc rotation angle
                rotate_angle= self.angle_of_line(a[0], a[1], b[0], b[1])

                # Rotation aug
                transform = A.Compose([
                    A.Rotate(
                        limit=(-rotate_angle, -rotate_angle), 
                        interpolation=cv2.INTER_LANCZOS4, 
                        always_apply=True,
                        ),
                ], keypoint_params= A.KeypointParams(format='xy', remove_invisible=False),
                )

                # Rotate Keypoints
                t= transform(image=img_copy, keypoints=[a,b])
                img_copy= t["image"]
                a,b= t["keypoints"]

                # Crop between keypoints
                img_copy= self.crop_around_keypoint(img_copy, b, y_crop_pct, x_crop_pct, series_description)

                # if idx == 0:
                #     print(img_copy.shape)
                #     import matplotlib.pyplot as plt
                #     plt.imshow(img_copy[:, :, img_copy.shape[-1]//2], cmap="gray")
                #     plt.axis("off")
                #     plt.savefig(f"img_{series_id}_{self.t2_width_pct}_{self.t2_offset_pct}_{self.t2_height_mult}.jpg", bbox_inches='tight', pad_inches=0)
                #     plt.close()

            # S1 Crop
            elif series_description == "Sagittal T1":
                # Mean between keypoints
                keypoint= ((a[0]+b[0])/2, (a[1]+b[1])/2)

                # Crop around keypoint
                img_copy= self.crop_around_keypoint(img_copy, keypoint, y_crop_pct, x_crop_pct, series_description)

                # if idx == 0:
                #     print(img_copy.shape)
                #     import matplotlib.pyplot as plt
                #     plt.imshow(img_copy[:, :, img_copy.shape[-1]//2])
                #     plt.savefig(f"img_{series_id}_{idx}_{self.t1_width_mult}_{self.t1_height_mult}.jpg", bbox_inches='tight', pad_inches=0)
                #     plt.close()

            else:
                raise ValueError("series_description not recognized.")
            
            # Resize + Norm
            img_copy= self.resize(image=img_copy)["image"]
            img_copy= self.percentile_norm(img_copy, percentiles)
            imgs.append(img_copy)

        # Stack
        imgs= np.stack(imgs, axis=2, dtype=np.uint8)
        return imgs

    def process_single_series(self, study_id, series_id, series_description):
        # Load img
        img = self.load_dcm(
            study_id= study_id, 
            series_id= series_id, 
            series_description= series_description,
            )

        # Save
        fpath= os.path.join(self.out_dir, str(study_id), str(series_id) + ".npy")
        np.save(fpath, img)

        return

    def process_study(self, x):
        study_id, series_md= x

        # Only processing sagittals
        for val in series_md:   
            self.process_single_series(
                    study_id= study_id, 
                    series_id= val["series_id"], 
                    series_description= val["series_description"],
                    )

        return study_id

    def run(self, ):
        # Load metadata
        mp_arr = list(self.df.values)
        print("n_dcms:", len(mp_arr))

        # Set core count
        cores= mp.cpu_count()
        print(f"Running on {cores} cores.")
        pool = mp.Pool(cores)
        
        # Run multiprocessing
        results= []
        for r in tqdm(pool.imap_unordered(self.process_study, mp_arr), total=len(mp_arr)):
            results.append(r)
            pass

        pool.close()
        pool.join()
        return

if __name__ == "__main__":

    # Load data
    df= pd.read_csv("./data/raw/train_series_descriptions.csv")
    df= df[~df.series_id.isin([
        3892989905, 2097107888, 2679683906, 1771893480, 
        996418962, 1753543608, 1848483560, # Bad Axial T2s
        ])]
    df= df[~df.study_id.isin([2492114990, 2780132468, 3008676218, 3637444890])]
    df= df[df.series_description != "Axial T2"]

    # Format for processing
    df= df.sort_values("series_description", ascending=False).reset_index(drop=True)
    df= df.groupby('study_id')[['series_id', 'series_description']].apply(lambda x: x.to_dict('records')).reset_index()
    print(df.iloc[0].values)
    print(df.head())
    
    # Stage 1: Coordinate Localization
    p= RsnaProcessor(
        df= df,
        stage = 1,
        in_dir="./data/raw/", 
        out_dir="./data/processed_stage1/",
        mode="train",
        )
    p.run()

    # Stage 2: Classification
    coords_sag= pd.read_csv("./data/metadata/coords_v7.csv")
    p= RsnaProcessor(
        df= df,
        coords_sag= coords_sag,
        stage = 2,
        in_dir="./data/raw/", 
        out_dir="./data/processed_stage2/",
        mode="train",
        )
    p.run()