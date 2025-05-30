#!/bin/bash
#BSUB -J task6
#BSUB -q hpc
#BSUB -o out/task5_n8_%J.out        # Standard output
#BSUB -e err/task5_n8_%J.err        # Standard error
#BSUB -W 100                         # Max wall time: 10 minutes
#BSUB -n 8                          # 1 CPU core
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=2000]"         # 2 GB RAM
#BSUB -R "select[model==XeonGold6126]" 
source /dtu/projects/02613_2025/conda/conda_init.sh

conda activate 02613

echo "=== Running for N=100 ==="
time python parallel_simulation_dynamic.py 100
