#!/bin/sh
#SBATCH -p gpu
#SBATCH -n1
#SBATCH --gres gpu:1
#SBATCH --time=0-12:00:00
 
export PATH=$HOME/miniconda3/bin:$PATH
export CONDA_PREFIX=$HOME/miniconda3/envs/kursinis
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
. activate base
conda activate kursinis

python3 main_timegan.py --data_name gear_signals --seq_len 2560 --module gru --hidden_dim 24 --num_layer 3 --iteration 3000 --batch_size 128 --metric_iteration 10 --feature_number Feat3
