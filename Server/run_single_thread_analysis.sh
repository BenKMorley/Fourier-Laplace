#!/bin/bash
#SBATCH --job-name=Fourier-Laplace    # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bkm1n18@soton.ac.uk    # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --time=36:00:00               # Time limit hrs:min:sec
#SBATCH --output=Server/log/Fourier-Laplace-%A.out
#SBATCH --error=Server/log/Fourier-Laplace-%A.err
#SBATCH -A dirac-dp099-cpu          # Which project to charge
#SBATCH -p icelake

N=$1
g=$2
m=$3
L=$4
T1=$5
T2=$6
CONFIGS=$7

# Source my virtual environment
source /home/dc-kitc1/virtual_envs/Fourier-Laplace/bin/activate

echo about to run python3 Server/run_3D_analysis.py $N $g $m $L \'$T1\' \'$T2\' $CONFIGS

python3 Server/run_3D_analysis.py $N $g $m $L \'$T1\' \'$T2\' $CONFIGS
