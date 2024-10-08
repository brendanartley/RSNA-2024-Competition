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

def save_prediction_img(cfg, batch, output, epoch):
    """
    Saves predictions ontop of image for visualization.
    """    
    in_shape= batch["input"].shape

    # Create fig
    fig, big_axes = plt.subplots(4,4, figsize=(20, 20))
    big_axes= big_axes.ravel()
    for i, ax in enumerate(big_axes):

        img = batch["input"][i, 0, 0, :, :].cpu().numpy() * 255
        cs_true= batch["target"][i, :].cpu().numpy() * cfg.img_size
        cs_pred= output["logits"][i, :].cpu().numpy() * cfg.img_size

        # Plot image
        ax.imshow(img, cmap='gray')
        ax.grid()
        ax.axis('off')

        # Plot coords + labels
        ax.plot(cs_true[:1], cs_true[1:], marker='o', c="tab:blue", markersize=10)
        ax.plot(cs_pred[:1], cs_pred[1:], marker='o', c="tab:orange", markersize=10)

    # Remove whitespace
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout(pad=0)
    plt.savefig(f'EPOCH_{epoch}_FOLD_{cfg.fold}_{cfg.plane}.jpg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return

def run_eval(model, val_ds, val_dl, val_metrics, cfg, epoch: int = 0, inference_run: bool = False):
    if len(val_ds) == 0:
        return {}
    model.eval()

    progress_bar = tqdm(range(len(val_dl)), disable=cfg.no_tqdm)
    val_itr = iter(val_dl)
    val_acc= 0
    i= 0
    logits= []
    targets= []
    idxs= []

    with torch.no_grad():
        for itr in progress_bar:
            i+=1
            data= next(val_itr)
            batch = batch_to_device(data, cfg.device, skip_keys=["idx"])

            if cfg.mixed_precision:
                with autocast():
                    output= model(batch)
            else:
                output= model(batch)

            logits.append(output["logits"])
            targets.append(batch["target"])
            
            if cfg.plane == "a2" and inference_run == True:
                idxs.append(batch["series_id"].repeat(len(output["logits"])))
            else:
                idxs.append(batch["series_id"])

            # Save from first validation batch
            if i == 1 and cfg.save_val_image and epoch == cfg.epochs-1 and inference_run == False:
                save_prediction_img(cfg, batch, output, epoch)

    logits= torch.cat(logits, dim=0).float()
    targets= torch.cat(targets, dim=0).float()
    idxs= torch.cat(idxs, dim=0).long()

    if inference_run or cfg.save_preds:
        # NOTE: Only works in test mode..
        if cfg.plane == "a2":

            # Move to CPU
            logits= logits.cpu()
            targets= targets.cpu()
            idxs= idxs.cpu()

            # Convert to pd.DataFrame() for kaggle metric
            submission= pd.DataFrame(
                logits,
                columns=["x", "y"],
                )
            submission["series_id"]= idxs

            submission["frame_index"]= submission.groupby("series_id").cumcount()
            submission= submission.sort_values(["series_id", "frame_index"]).reset_index(drop=True)

            if inference_run:
                return submission

            if cfg.save_preds:
                outpath= "./data/preds/{}_{}_{}_{}.csv".format("sub", cfg.fold, cfg.config_file, cfg.seed)
                submission.to_csv(outpath, index=False)


        else:
            # Save preds
            logits= logits.reshape(-1, 2)
            targets= targets.reshape(-1, 2)

            # Create other col values
            levels= ["L1/L2","L1/L2","L2/L3","L2/L3","L3/L4","L3/L4","L4/L5","L4/L5","L5/S1","L5/S1"] * idxs.shape[0]
            sides= ["L","R"] * 5 
            sides= sides * idxs.shape[0]
            idxs= torch.repeat_interleave(idxs, 10)

            # Move to CPU
            logits= logits.cpu()
            targets= targets.cpu()
            idxs= idxs.cpu()

            # Convert to pd.DataFrame() for kaggle metric
            submission= pd.DataFrame(
                logits,
                columns=["relative_x", "relative_y"],
                )
            submission["series_id"]= idxs
            submission["level"]= levels
            submission["side"]= sides
            submission= submission.groupby(['series_id', 'level', 'side'])[['relative_x', 'relative_y']].mean().reset_index()
            submission= submission.sort_values(['series_id', 'level', 'side'])

            if inference_run:
                return submission

            if cfg.save_preds:
                outpath= "./data/preds/{}_{}_{}_{}.csv".format("sub", cfg.fold, cfg.config_file, cfg.seed)
                submission.to_csv(outpath, index=False)

    if not inference_run:
        # Metrics
        val_metrics["loss"]= model.loss_fn(logits, targets).item()

        # Custom Metrics
        # PCT of pixels w/in range of true value
        for pct in [1,5]:
            acc= torch.abs((logits - targets)) < (pct/100)
            acc= acc.float().mean().item()
            val_metrics["within_{}_pct".format(pct)]= acc

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
    epoch= 0
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
                batch= batch_to_device(data, cfg.device, skip_keys=["idx"])

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
                if cfg.fast_dev_run == True or (cfg.local_rank == 0 and i % cfg.logging_steps == 0):
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
                    val_metrics= run_eval(model, val_ds, val_dl, val_metrics, cfg, epoch=epoch)

                    # Progress bar
                    progress_bar.set_postfix(flatten_dict(train_metrics | val_metrics))

                    # Logger
                    logger.log({"val": val_metrics})

    # Final Eval
    val_metrics= run_eval(model, val_ds, val_dl, val_metrics, cfg, epoch=epoch)
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