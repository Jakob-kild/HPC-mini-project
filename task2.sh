#!/bin/bash
#BSUB -J task2
#BSUB -q hpc
#BSUB -o out/task2%J.out        # Standard output
#BSUB -e err/task2%J.err        # Standard error
#BSUB -W 10                         # Max wall time: 10 minutes
#BSUB -n 1                          # 1 CPU core
#BSUB -R "rusage[mem=2000]"         # 2 GB RAM
#BSUB -R "select[model==XeonGold6126]" 


echo "=== Running for N=10 ==="
time python simulation.py 3


