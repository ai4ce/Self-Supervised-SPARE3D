#!/bin/bash


train_path=/data_1/yfx/SPARE3D/Data/Three2I_filt_Data_final/train

valid_path=/data_1/yfx/SPARE3D/Data/Three2I_filt_Data_final/valid

out_path=/data_1/yfx/SPARE3D/CVPR_Jigsaw/train_out1
pre_train=/data_1/yfx/SPARE3D/CVPR_Jigsaw/jps_010_117000.pth

BS=16

LR=0.00001

model=vgg16


CUDA_VISIBLE_DEVICES=3 python ThreeV2I_BC_trainer.py --Training_dataroot $train_path --Validating_dataroot $valid_path --batchSize $BS --lr $LR --model_type $model --outf $out_path --pretrain_dataroot $pre_train


LR=0.000005
out_path=/data_1/yfx/SPARE3D/CVPR_Jigsaw/train_out2

CUDA_VISIBLE_DEVICES=3 python ThreeV2I_BC_trainer.py --Training_dataroot $train_path --Validating_dataroot $valid_path --batchSize $BS --lr $LR --model_type $model --outf $out_path --pretrain_dataroot $pre_train


LR=0.000001
out_path=/data_1/yfx/SPARE3D/CVPR_Jigsaw/train_out3
CUDA_VISIBLE_DEVICES=3 python ThreeV2I_BC_trainer.py --Training_dataroot $train_path --Validating_dataroot $valid_path --batchSize $BS --lr $LR --model_type $model --outf $out_path --pretrain_dataroot $pre_train

