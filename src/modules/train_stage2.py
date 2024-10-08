import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import numpy as np
import os

from tqdm import tqdm
import gc

from src.data.utils import (
    get_data,
    get_dataset,
    get_dataloader,
)

from src.modules.utils import (
    get_model,
    get_optimizer,
    get_scheduler,
    batch_to_device,
    calc_grad_norm,
    flatten_dict,
    save_weights,
)

from src.logging.utils import (
    get_logger,
)

from src.metrics.metric import (
    score, single_score,
)

def run_eval(model, val_ds, val_dl, val_metrics, cfg, inference_run: bool = False):
    if len(val_ds) == 0:
        return {}
    model.eval()

    progress_bar = tqdm(range(len(val_dl)), disable=cfg.no_tqdm)
    val_itr = iter(val_dl)
    val_acc= 0
    i= 0
    logits= []
    targets= []
    study_ids= []
    ttas= []

    with torch.no_grad():
        for itr in progress_bar:
            i+=1

            data= next(val_itr)
            batch = batch_to_device(data, cfg.device)

            if cfg.mixed_precision:
                with autocast():
                    output = model(batch)
            else:
                output = model(batch)

            logits.append(output["logits"])
            targets.append(batch["target"])
            study_ids.append(batch["study_id"])
            
            if inference_run:
                ttas.append(batch["tta"])

    logits= torch.cat(logits, dim=0).float()
    logits= F.softmax(logits, dim=1)
    targets= torch.cat(targets, dim=0)
    study_ids= torch.cat(study_ids, dim=0)

    # TTA Flip Labels
    if inference_run:
        logits= logits.view(-1,3)
        targets= targets.view(-1,3)
    else:
        targets= targets.view(-1,3)
        val_metrics["loss"]= model.loss_fn(logits, targets).item()

    # ---- Kaggle Metric ----
    study_ids= study_ids.repeat_interleave(cfg.n_classes//3)

    # OHE targets
    mask= (targets[:, 0] == -100).bool().cpu()

    # Move to CPU
    logits= logits.cpu()
    targets= targets.cpu()

    # Create row_ids
    row_ids= []
    for i, study_id in enumerate(study_ids):
        row_id= "{}_{}".format(study_id, val_ds.label_cols[i%(cfg.n_classes//3)])
        row_ids.append(row_id)

    # Class weights
    class_weights= torch.tensor([1.0, 2.0, 4.0])
    sample_weights = torch.full_like(targets[:, 0], fill_value=-100, dtype=torch.float)
    sample_weights[~mask] = class_weights[torch.argmax(targets, dim=1)[~mask]]

    # Convert to pd.DataFrame() for kaggle metric
    submission= pd.DataFrame(
        logits,
        columns=['normal_mild', 'moderate', 'severe'],
        )
    submission["row_id"]= row_ids
    submission["mask"]= mask
    submission= submission.groupby("row_id").mean().reset_index()
    submission= submission.sort_values("row_id")

    if inference_run:
        submission= submission[submission["mask"] == 0].drop(columns=["mask"]).reset_index(drop=True)
        return submission

    solution= pd.DataFrame(
        targets,
        columns=['normal_mild', 'moderate', 'severe'],
        )
    solution["row_id"]= row_ids
    solution["sample_weight"]= sample_weights
    solution["mask"]= mask
    solution= solution.groupby("row_id").mean().reset_index()
    solution= solution.sort_values("row_id")

    # Save preds
    if cfg.save_preds:
        outpath= "./data/preds/{}_{}_{}_{}.csv".format("{}", cfg.fold, cfg.config_file, cfg.seed)
        submission.drop(columns=["mask"]) \
                  .to_csv(outpath.format("sub"), index=False)

    # Remove mask col
    submission= submission[submission["mask"] == 0].drop(columns=["mask"]).reset_index(drop=True)
    solution= solution[solution["mask"] == 0].drop(columns=["mask"]).reset_index(drop=True)
    # print(submission.shape, solution.shape)

    # Score preds
    if cfg.condition == "all":
        kaggle_metrics= score(solution= solution, submission= submission, row_id_column_name="row_id", any_severe_scalar=1.0)
    else:
        kaggle_metrics= single_score(solution= solution, submission= submission, row_id_column_name="row_id", any_severe_scalar=1.0)

    val_metrics= val_metrics | kaggle_metrics

    return val_metrics

def train(cfg):

    # Used when n_gpus > 1
    cfg.local_rank= 0

    # Load logger
    logger= get_logger(cfg)

    # Data
    df, val_df= get_data(cfg)

    train_ds= get_dataset(df, cfg, mode="train")
    train_dl= get_dataloader(train_ds, cfg, mode="train")

    val_ds= get_dataset(val_df, cfg, mode="val")
    val_dl= get_dataloader(val_ds, cfg, mode="val")

    # Model/optimizer/scheduler
    model, n_params = get_model(cfg)
    logger.log({"n_params": n_params}, commit=False)
    model.to(cfg.device)

    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg, n_steps=len(train_dl)*cfg.epochs)

    if cfg.mixed_precision:
        scaler = GradScaler()
    else:
        scaler = None

    # Training Loop
    train_metrics= {"lr": None, "epoch": None}
    val_metrics= {}
    i= 0
    total_grad_norm = None    
    total_grad_norm_after_clip = None
    optimizer.zero_grad()

    if cfg.train:
        for epoch in range(cfg.epochs):
            losses= []
            grad_norms= []
            grad_norms_clipped= []
            gc.collect()

            if cfg.local_rank == 0: 
                train_metrics["epoch"] = epoch

            progress_bar = tqdm(range(len(train_dl)), disable=cfg.no_tqdm)

            try:
                tr_itr = iter(train_dl)
            except Exception as e:
                print(e)
                print("BATCH FETCH FAILED.")

            for itr in progress_bar:
                i += 1
                data= next(tr_itr)
                batch= batch_to_device(data, cfg.device)

                model= model.train()

                # Forward Pass
                if cfg.mixed_precision:
                    with autocast():
                        output= model(batch)
                else:
                    output= model(batch)
                loss= output["loss"]
                losses.append(loss.item())

                # Backward pass
                if cfg.mixed_precision:
                    scaler.scale(loss).backward()
                    if i % cfg.grad_accumulation == 0:
                        if (cfg.track_grad_norm) or (cfg.grad_clip > 0):
                            scaler.unscale_(optimizer)
                        if cfg.track_grad_norm:
                            total_grad_norm = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
                            if total_grad_norm is not None:
                                grad_norms.append(total_grad_norm.item())
                        if cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        if cfg.track_grad_norm:
                            total_grad_norm_after_clip = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
                            if total_grad_norm_after_clip is not None:
                                grad_norms_clipped.append(total_grad_norm_after_clip.item())
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    if i % cfg.grad_accumulation == 0:
                        if cfg.track_grad_norm:
                            total_grad_norm = calc_grad_norm(model.parameters())
                            if total_grad_norm is not None:
                                grad_norms.append(total_grad_norm.item())
                        if cfg.grad_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                        if cfg.track_grad_norm:
                            total_grad_norm_after_clip = calc_grad_norm(model.parameters(), cfg.grad_norm_type)
                            if total_grad_norm_after_clip is not None:
                                grad_norms_clipped.append(total_grad_norm_after_clip.item())
                        optimizer.step()
                        optimizer.zero_grad() 

                if scheduler is not None:
                    scheduler.step()

                # Train Logging
                if cfg.local_rank == 0 and i % cfg.logging_steps == 0:
                    train_metrics["loss"]= np.mean(losses[-10:])
                    train_metrics["lr"]= cfg.lr if scheduler is None else scheduler.get_last_lr()[0]

                    if cfg.track_grad_norm:
                        train_metrics["grad_norm"] = np.mean(grad_norms[-10:])
                        train_metrics["grad_norm_clipped"] = np.mean(grad_norms_clipped[-10:])
            
                    # Progress bar
                    progress_bar.set_postfix(flatten_dict(train_metrics | val_metrics))

                    # Logger
                    logger.log({"train": train_metrics})
                    pass

                # Eval Logging
                if cfg.local_rank == 0 and i % cfg.eval_steps == 0:
                    val_metrics= run_eval(model, val_ds, val_dl, val_metrics, cfg)

                    # Progress bar
                    progress_bar.set_postfix(flatten_dict(train_metrics | val_metrics))

                    # Logger
                    logger.log({"val": val_metrics})

    # Final eval
    val_metrics= run_eval(model, val_ds, val_dl, val_metrics, cfg)
    logger.log({"val": val_metrics})
    print(val_metrics)

    # Save weights
    if cfg.save_weights:
        save_weights(
            model, 
            cfg, 
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            )

    logger.finish()
    return