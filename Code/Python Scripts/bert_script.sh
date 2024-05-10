#!/bin/bash
#
#SBATCH --nodelist=nlpgpu02
#SBATCH --gpus=1
#SBATCH --partition=p_nlp
#
#SBATCH --ntasks=1
#SBATCH --mem=256GB

srun python BERTopic_css.py