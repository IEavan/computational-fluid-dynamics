#!/bin/bash
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=3882
#SBATCH --time=00:01:00
#SBATCH --partition=cs402
module purge
module load intel impi
srun --output output ./karman --verbose=2 --infile=initial.bin
