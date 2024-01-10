#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu:1
#SBATCH --time=0-10:00:00
 
export PATH=$HOME/miniconda3/bin:$PATH
export CONDA_PREFIX=$HOME/miniconda3/envs/kursinis
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
. activate base
conda activate kursinis

python3 main_timegan.py --do_training yes --model_dir models_feat3_gru_6dim_48lay --data_name gear_signals --seq_len 32 --dataset_percentage 25 --module gru --hidden_dim 48 --num_layer 6 --iteration 3000 --batch_size 128 --metric_iteration 1 --feature_number Feat3
