#!/bin/bash
 
#wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#bash Miniconda3-latest-Linux-x86_64.sh
# yes - on install, no - in init, path default one
 
export PATH=~/miniconda3/bin:$PATH
#conda create -n kursinis -c conda-forge tensorflow-gpu=1.15 python=3.7
. activate base
conda activate kursinis
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

#pip3 install -r requirements.txt

