#!/bin/bash
#SBATCH -J SAM3D_TFG
#SBATCH -p short
#SBATCH -N 1
#SBATCH -w node028
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --chdir=/home/ematas/tfg
#SBATCH -o slurm.SAM3D.%j.out
#SBATCH -e slurm.SAM3D.%j.err

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYOPENGL_PLATFORM=egl   

module load Miniconda3/4.9.2
module load CUDA/12.1

eval "$(conda shell.bash hook)"
conda activate sam3d_env

cd sam-3d-body

echo "=== START SAM3D ==="
nvidia-smi

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=0 

python -m nbconvert --to notebook --execute ../Pruebas_SAM3D_GPU_issues_Pulldown.ipynb --output Resultado_SAM3D.ipynb

echo "=== EXECUTION COMPLETED ==="
