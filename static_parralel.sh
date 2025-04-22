write a script that has a for loop that changes the n of cores, its .sh script:

#!/bin/bash
#BSUB -J task5
#BSUB -q hpc
#BSUB -o out/task5%J.out        # Standard output
#BSUB -e err/task5%J.err        # Standard error
#BSUB -W 10                         # Max wall time: 10 minutes
#BSUB -n 1                          # 1 CPU core
#BSUB -R "rusage[mem=2000]"         # 2 GB RAM
#BSUB -R "select[model==XeonGold6126]" 
source /dtu/projects/02613_2025/conda/conda_init.sh

conda activate 02613

echo "=== Running for N=100 ==="
time python parralel.py 100

