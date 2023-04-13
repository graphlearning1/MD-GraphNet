#!/bin/bash
#SBATCH --gpus=1
module load anaconda/2022.11
source activate py37
python run_opt.py