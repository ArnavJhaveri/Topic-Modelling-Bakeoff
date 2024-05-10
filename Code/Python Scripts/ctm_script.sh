#!/bin/bash
#
#SBATCH --nodelist=nlpgpu03
#SBATCH --gpus=1
#SBATCH --partition=p_nlp
#
#SBATCH --ntasks=1
#SBATCH --mem=256GB

srun python CTM_css.py