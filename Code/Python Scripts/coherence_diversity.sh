#!/bin/bash
#
#SBATCH --nodelist=nlpgpu01
#SBATCH --gpus=1
#SBATCH --partition=p_nlp
#
#SBATCH --ntasks=1
#SBATCH --mem=128GB

srun python coherence_diversity.py