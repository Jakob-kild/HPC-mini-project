#!/bin/bash
#BSUB -J task9
#BSUB -q c02613
#BSUB -o out/task9_n1_%J.out        # Standard output
#BSUB -e err/task9_n1_%J.err        # Standard error
#BSUB -W 00:30                         # Max wall time: 10 minutes
#BSUB -n 4                          # 1 CPU core
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1GB]"         # 2 GB RAM
#BSUB -gpu "num=1:mode=exclusive_process"  # request 1 GPU

source /dtu/projects/02613_2025/conda/conda_init.sh

conda activate 02613

echo "=== Running for N=10 ==="
python ex10.py 4571 > results.csv

