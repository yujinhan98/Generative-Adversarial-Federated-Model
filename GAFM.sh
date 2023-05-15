#!/bin/bash

#SBATCH --job-name=G_ISIC
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=output_ISIC_G_Test.txt
#SBATCH --time=2-

#module load miniconda3
#module load cuDNN/7.6.2.24-CUDA-10.0.130
#module load CUDA
#module load cuDNN
#module load /home/yh579/miniconda3
#module load cuDNN/7.6.2.24-CUDA-10.0.130 #/home/yh579/miniconda3
#load python/3.9.12
#conda activate YJ
python train.py










