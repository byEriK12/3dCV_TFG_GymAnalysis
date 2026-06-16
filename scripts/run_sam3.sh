#!/bin/bash
#SBATCH -J SAM3_TFG_FINAL  
#SBATCH -p short               
#SBATCH -N 1                   
#SBATCH -w node028          
#SBATCH --gres=gpu:l40s:1      
#SBATCH --mem=200G              
#SBATCH --cpus-per-task=16     
#SBATCH --time=01:59:00        
#SBATCH --chdir=/home/ematas/tfg
#SBATCH -o slurm.FINAL.%j.out   
#SBATCH -e slurm.FINAL.%j.err    

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load Miniconda3/4.9.2
module load CUDA/12.1

eval "$(conda shell.bash hook)"
conda activate sam3_env

echo "--- START SAM3 ---"
date
nvidia-smi

jupyter nbconvert --to notebook --execute Pruebas_SAM3_GPU_issues.ipynb --output Resultado_TFG_Final.ipynb

echo "--- EXECUTION COMPLETED ---"
date
