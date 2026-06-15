#!/bin/bash
#SBATCH -J SAM3_TFG_FINAL        # Nombre del trabajo
#SBATCH -p short                 # Partición
#SBATCH -N 1                     # 1 solo nodo
#SBATCH -w node028               # <--- FORZAMOS EL NODO DE 48GB (L40S)
#SBATCH --gres=gpu:l40s:1        # Pedimos 1 GPU L40S de ese nodo
#SBATCH --mem=200G               # Memoria RAM del sistema
#SBATCH --cpus-per-task=16       # Núcleos de CPU
#SBATCH --time=01:59:00          # Tiempo máximo (ajusta si tu vídeo es muy largo)
#SBATCH --chdir=/home/ematas/tfg
#SBATCH -o slurm.FINAL.%j.out    # Archivo de salida con ID del job
#SBATCH -e slurm.FINAL.%j.err    # Archivo de errores

# Optimización para evitar fragmentación de memoria
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Carga de módulos
module load Miniconda3/4.9.2
module load CUDA/12.1

# Activar entorno Conda
eval "$(conda shell.bash hook)"
conda activate sam3_env

echo "--- INICIO DE EJECUCIÓN EN NODO POTENTE ---"
date
nvidia-smi

# Ejecutar el notebook y generar uno nuevo con los resultados
jupyter nbconvert --to notebook --execute Pruebas_SAM3_GPU_issues.ipynb --output Resultado_TFG_Final.ipynb

echo "--- TRABAJO FINALIZADO CON ÉXITO ---"
date
