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

python3 vizualize_data.py