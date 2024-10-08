#!/bin/bash

# example command:
# nohup ./run1.sh > nohup1.out &
# nohup ./run2.sh > nohup2.out &

python train.py -G=0 -C=cfg_stage2_s1_fo grad_checkpointing=True save_weights=True fold=0
python train.py -G=0 -C=cfg_stage2_s2_su grad_checkpointing=True save_weights=True fold=0
python train.py -G=0 -C=cfg_stage2_s2_sp1 grad_checkpointing=True save_weights=True fold=0
python train.py -G=0 -C=cfg_stage2_s2_sp2 grad_checkpointing=True save_weights=True fold=0

python train.py -G=0 -C=cfg_stage2_s1_fo grad_checkpointing=True save_weights=True fold=1
python train.py -G=0 -C=cfg_stage2_s2_su grad_checkpointing=True save_weights=True fold=1
python train.py -G=0 -C=cfg_stage2_s2_sp1 grad_checkpointing=True save_weights=True fold=1
python train.py -G=0 -C=cfg_stage2_s2_sp2 grad_checkpointing=True save_weights=True fold=1

python train.py -G=0 -C=cfg_stage2_s1_fo grad_checkpointing=True save_weights=True fold=2
python train.py -G=0 -C=cfg_stage2_s2_su grad_checkpointing=True save_weights=True fold=2
python train.py -G=0 -C=cfg_stage2_s2_sp1 grad_checkpointing=True save_weights=True fold=2
python train.py -G=0 -C=cfg_stage2_s2_sp2 grad_checkpointing=True save_weights=True fold=2

python train.py -G=0 -C=cfg_stage2_s1_fo grad_checkpointing=True save_weights=True fold=3
python train.py -G=0 -C=cfg_stage2_s2_su grad_checkpointing=True save_weights=True fold=3
python train.py -G=0 -C=cfg_stage2_s2_sp1 grad_checkpointing=True save_weights=True fold=3
python train.py -G=0 -C=cfg_stage2_s2_sp2 grad_checkpointing=True save_weights=True fold=3