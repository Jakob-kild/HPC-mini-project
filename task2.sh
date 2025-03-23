#!/bin/bash
#BSUB -J task2
#BSUB -q hpc
#BSUB -o task2.out        # Standard output
#BSUB -e task2.err        # Standard error
#BSUB -W 10                         # Max wall time: 10 minutes
#BSUB -n 1                          # 1 CPU core
#BSUB -R "rusage[mem=2000]"         # 2 GB RAM
#BSUB -R "select[model==XeonGold6126]" 
#BSUB -B                            # Email when job starts
#BSUB -N                            # Email when job ends
#BSUB -u s214968@dtu.dk             # Email address

# Activate conda
# source /dtu/projects/02613_2025/conda/conda_init.sh

echo "=== Running for N=10 ==="
time python simulation.py 10

echo "=== Running for N=15 ==="
time python simulation.py 15

echo "=== Running for N=20 ==="
time python simulation.py 20
