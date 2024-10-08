from types import SimpleNamespace
import torch
import socket
import os

os.environ['NO_ALBUMENTATIONS_UPDATE']= "1"

import albumentations as A
import cv2

# General
cfg = SimpleNamespace(**{})
cfg.train_df= "./data/metadata/metadata.csv"
cfg.img_dir= "./data/processed_stage2/"
cfg.project= "rsna"
cfg.hostname = socket.gethostname()
cfg.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.weights_path= ""
cfg.overfit_batch= False
cfg.fast_dev_run= False
cfg.save_weights= False
cfg.save_preds= False
cfg.logger= None # None, "wandb"
cfg.no_tqdm= False
cfg.train= True
cfg.val= True
cfg.n_classes= 30
cfg.seed= -1
cfg.fold= 0
cfg.ignore_index= -100

# RSNA specific
cfg.plane= "s1" # "s1", "s2", "a2"
cfg.condition= "foraminal" # "foraminal", "subarticular", "spinal", "all"

# Optimizer + Scheduler
cfg.scheduler = "CosineAnnealingLR" # Constant, CosineAnnealingLR
cfg.lr = 1e-4
cfg.lr_min= 1e-6
cfg.weight_decay = 1e-4
cfg.epochs= 32

# Dataset/Dataloader
cfg.num_workers= 4
cfg.drop_last= True
cfg.pin_memory = False
cfg.data_sample= 0
cfg.batch_size= 32
cfg.batch_size_val= None

# Model
cfg.model_type = "cnn25d_sagittal_1head"
cfg.backbone= "hgnetv2_b5.ssld_stage2_ft_in1k"
cfg.drop_path_rate= 0.2
cfg.fc_dropout= 0.05
cfg.in_chans= 1
cfg.attn_type= "AttentionLSTM"
cfg.n_frames= 24

# Augs
img_size= 96
cfg.img_size= img_size
cfg.n_tta= 1
cfg.cutmix_prob= 0.3
cfg.mixup_prob= 0.3
cfg.mixup_alpha= 1.0
cfg.train_aug= A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4), contrast_limit=(-0.25, 0.25), p=1.0),
    A.ShiftScaleRotate(
        always_apply=False,
        p=1.0,
        shift_limit_x=(-0.15, 0.15),
        shift_limit_y=(-0.15, 0.15),
        scale_limit=(-0.25, 0.25),
        rotate_limit=(-30, 30),
        interpolation=1,
        border_mode=4,
    ),
    A.OneOf([
        A.MotionBlur(blur_limit=3),
        A.MedianBlur(blur_limit=3),
        A.GaussianBlur(blur_limit=3),
        A.GaussNoise(var_limit=(3.0, 9.0)),
    ], p=0.5),
    A.OneOf([
        A.OpticalDistortion(distort_limit=1.),
        A.GridDistortion(num_steps=5, distort_limit=1.),
    ], p=0.5),
    A.CoarseDropout(
        max_holes=250,
        max_height=1,
        max_width=1,
        min_holes=50,
        fill_value=0,
        p=0.5,
    ),
    A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LANCZOS4, always_apply=True),
    A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), always_apply=True),
])

cfg.val_aug= A.Compose([
    A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LANCZOS4, always_apply=True),
    A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(0, 0, 0), always_apply=True),
])

# Other
cfg.mixed_precision= True
cfg.grad_checkpointing= False
cfg.grad_accumulation= 1
cfg.track_grad_norm= True
cfg.grad_clip= 1.0
cfg.grad_norm_type= 2
cfg.logging_steps= 25
cfg.eval_steps= 50