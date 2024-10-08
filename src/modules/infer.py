import pandas as pd
import os
os.environ['NO_ALBUMENTATIONS_UPDATE']= "1"

import numpy as np
import glob
import torch
import json
from copy import deepcopy
from typing import Iterable
from types import SimpleNamespace

from importlib import import_module

from src.preprocess.rsna import RsnaProcessor

from src.data.dataset_stage1 import Stage1Dataset
from src.data.dataset_stage2 import Stage2Dataset

from src.data.utils import get_val_dataloader

from src.modules.train_stage1 import run_eval as run_eval_stage1
from src.modules.train_stage2 import run_eval as run_eval_stage2

from src.modules.utils import get_model

def run_preds(
        df: pd.DataFrame,
        config_file: str = "",
        stage: int = 1,
        wpaths: Iterable[str] = [], 
        img_dir: str = "",
        n_tta: int = 1,
        fold: int = -100,
    ):
    cfg= import_module(f"src.configs.{config_file}").cfg

    # Config
    cfg.img_dir= img_dir
    cfg.n_tta= n_tta
    cfg.fold= fold

    # Filter DF
    plane_map= {"s2":"Sagittal T2/STIR", "s1":"Sagittal T1"}
    z= df[df.series_description == plane_map[cfg.plane]].copy()

    # Dataset/Dataloader
    if stage == 1:
        ds= Stage1Dataset(z, cfg, mode="test")
    elif stage == 2:
        ds= Stage2Dataset(z, cfg, mode="test")
    dl= get_val_dataloader(ds, cfg)

    # Run Inference
    for i, wpath in enumerate(wpaths):
        c= deepcopy(cfg)
        c.weights_path= wpath

        # Load metadata
        with open(wpath.replace(".pt", ".json"), 'r') as f:
            md= json.load(f)

        # Update config
        c.backbone= md["backbone"]
        c.attn_type= md["attn_type"]
        c.model_type= md["model_type"]
        c.seed= md["seed"]
        
        # Load model
        m, n_params= get_model(c, pretrained=False, inference_mode=True)
        m= m.eval().to(c.device)

        # Run eval loop
        if stage == 1:
            res= run_eval_stage1(m, ds, dl, {}, c, inference_run=True)
        elif stage == 2:
            res= run_eval_stage2(m, ds, dl, {}, c, inference_run=True)
            res["config_file"]= config_file

        # Save preds
        res.to_csv("{}_{}_{}.csv".format(config_file, c.seed, c.fold), index=False)

        # Cleanup
        del m
        torch.cuda.empty_cache()
    return

if __name__ == "__main__":
    # Inference run
    N_TTA= 9
    N_SAMPLES= 999999
    FOLDS= [0,1,2,3]
    RUN = True

    # Load models
    MODELS= {}
    for f in FOLDS:
        MODELS[f]= {
            "S2": {
                "cfg_stage1_s2": glob.glob("./data/checkpoints/cfg_stage1_s2_fold{}*.pt".format(f)),
                "cfg_stage2_s2_sp1": glob.glob("./data/checkpoints/cfg_stage2_s2_sp1_fold{}*.pt".format(f)),
                "cfg_stage2_s2_sp2": glob.glob("./data/checkpoints/cfg_stage2_s2_sp2_fold{}*.pt".format(f)),
                "cfg_stage2_s2_su": glob.glob("./data/checkpoints/cfg_stage2_s2_su_fold{}*.pt".format(f)),
            },
            "S1": {
                "cfg_stage1_s1": glob.glob("./data/checkpoints/cfg_stage1_s1_fold{}*.pt".format(f)),
                "cfg_stage2_s1_fo": glob.glob("./data/checkpoints/cfg_stage2_s1_fo_fold{}*.pt".format(f)),
            },
        }
    print(json.dumps(MODELS, indent=4))

    # Run process on each fold
    for FOLD in FOLDS:
        if not RUN:
            continue

        print("-"*25, f" RUNNING FOLD {FOLD} ", "-"*25)
        cols= ['study_id', 'series_id', 'series_description'] 
        df= pd.read_csv("./data/raw/train_series_descriptions.csv")
        md= pd.read_csv("./data/metadata/metadata.csv")

        # Select FOLD data
        if FOLD != -100:
            md= md.loc[(md.fold == FOLD), :]
        df= pd.merge(df[cols], md[cols], how="inner", on=cols).reset_index(drop=True)
        df_all = df.copy() # Save all to fill missing values
        
        # Remove bad IDxs
        df= df[~df.series_id.isin([
            3892989905, 2097107888, 2679683906, 1771893480, 
            996418962, 1753543608, 1848483560, # Bad Axial T2s
            ])]
        df= df[~df.study_id.isin([2492114990, 2780132468, 3008676218, 3637444890])]
        df= df[df.series_description != "Axial T2"]
        df= df.groupby("series_description").head(N_SAMPLES)

        # Select data
        df= df.sort_values("series_description", ascending=False).reset_index(drop=True)
        z= df.groupby('study_id')[['series_id', 'series_description']].apply(lambda x: x.to_dict('records')).reset_index().copy()
        
        # ---------------------------------------- Stage 1 ----------------------------------------
        # Process
        p= RsnaProcessor(
            df= z,
            stage = 1,
            in_dir="./data/raw/", 
            out_dir="./data/sample_stage1",
            mode="train",
            )
        p.run()

        # S1: Vertabrae Localization
        for config_file in ["cfg_stage1_s1"]:
            run_preds(
                df= df,
                config_file= config_file,
                stage= 1,
                wpaths= MODELS[FOLD]["S1"][config_file],
                img_dir= "./data/sample_stage1",
                fold= FOLD,
            )

        # S2: Vertabrae Localization
        for config_file in ["cfg_stage1_s2"]:
            run_preds(
                df= df,
                config_file= config_file,
                stage= 1,
                wpaths= MODELS[FOLD]["S2"][config_file],
                img_dir= "./data/sample_stage1",
                fold= FOLD,
            )

        # # ---------------------------------------- Stage 2 ----------------------------------------
        # Load SagT1s
        df1= pd.concat([pd.read_csv(x) for x in glob.glob("./cfg_stage1_s1*_{}.csv".format(FOLD))], axis=0)
        df1= df1.groupby(['series_id', 'level', 'side'])[['relative_x', 'relative_y']].median().reset_index()

        # Load SagT2s
        df2= pd.concat([pd.read_csv(x) for x in glob.glob("./cfg_stage1_s2*_{}.csv".format(FOLD))], axis=0)
        df2= df2.groupby(['series_id', 'level', 'side'])[['relative_x', 'relative_y']].median().reset_index()

        # Combine
        coords_sag= pd.concat([df1, df2], axis=0).reset_index(drop=True)
        coords_sag= coords_sag.sort_values(["series_id", "level", "side"]).reset_index(drop=True)
        coords_sag.to_csv(f"coords_{FOLD}.csv", index=False)

        coords_ax= None
        del df1, df2

        # Process
        p= RsnaProcessor(
            df= z,
            coords_sag= coords_sag,
            stage = 2,
            in_dir="./data/raw/", 
            out_dir=f"./data/sample_stage2",
            mode="train",
            )
        p.run()

        # S1: Classification
        for config_file in ["cfg_stage2_s1_fo"]:
            run_preds(
                df= df,
                config_file= config_file,
                stage= 2,
                wpaths= MODELS[FOLD]["S1"][config_file],
                img_dir= "./data/sample_stage2",
                fold= FOLD,
                n_tta= N_TTA,
            )

        # S2: Classification
        for config_file in ["cfg_stage2_s2_sp1", "cfg_stage2_s2_sp2", "cfg_stage2_s2_su"]:
            run_preds(
                df= df,
                config_file= config_file,
                stage= 2,
                wpaths= MODELS[FOLD]["S2"][config_file],
                img_dir= "./data/sample_stage2",
                fold= FOLD,
                n_tta= N_TTA,
            )

    # -------------- Merge Predictions ----------------
    arr= []

    # Load SagT1s
    fpaths= glob.glob("./cfg_stage2_s1*.csv")
    df1= pd.concat([pd.read_csv(x) for x in fpaths], axis=0)
    df1= df1.drop(columns=["config_file"])
    df1= df1.groupby("row_id").mean().reset_index()

    # Load SagT2s
    fpaths= glob.glob("./cfg_stage2_s2*.csv")
    df2= pd.concat([pd.read_csv(x) for x in fpaths], axis=0)
    df2= df2[~((df2["row_id"].str.contains("subarticular")) & (df2["config_file"].isin(["cfg_stage2_s2_sp1", "cfg_stage2_s2_sp2"])))]
    df2= df2.drop(columns=["config_file"])
    df2= df2.groupby("row_id").mean().reset_index()

    # Remove S2s foraminal preds when S1 is present
    df2= df2.loc[
        (~df2.row_id.isin(df1.loc[df1["row_id"].str.contains("foraminal")].row_id.values)) | \
        (~df2["row_id"].str.contains("foraminal")),
        :,
        ].copy()
    arr.extend([df1, df2])

    # Combine into sub
    sub= pd.concat(arr).reset_index(drop=True)
    sub= sub.groupby("row_id").mean().reset_index()
    sub= sub.sort_values("row_id").reset_index(drop=True)
        
    # -------------- Score DF ----------------
    # Remove rows that have no labels
    sol= pd.read_csv("./data/metadata/study_id_labels.csv")
    print(sol.shape, sub.shape)
    sol= pd.merge(sol, sub["row_id"], how="inner", on="row_id")
    sub= pd.merge(sub, sol["row_id"], how="inner", on="row_id")
    print(sol.shape, sub.shape)

    sol= sol.sort_values("row_id").reset_index(drop=True)
    sub= sub.sort_values("row_id").reset_index(drop=True)
    sub.to_csv("./data/bartley_sagittal_oof.csv", index=False)

    weights= np.array([1.0, 2.0, 4.0])
    sol["sample_weight"]= weights[np.argmax(sol[['normal_mild', 'moderate', 'severe']], axis=1)]

    from src.metrics.metric import score, single_score
    try:
        d= score(solution= sol.copy(), submission= sub.copy(), row_id_column_name="row_id", any_severe_scalar=1.0)
    except:
        d= single_score(solution= sol.copy(), submission= sub.copy(), row_id_column_name="row_id", any_severe_scalar=1.0)
    for k, v in d.items():
        print(k, round(v, 4))
