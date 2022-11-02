#!/bin/bash

NAME=Core/Binderanalysis.py

N=$1
g=$2
m=$3
L=$4
T1=$5
T2=$6

# Source my virtual environment
source /home/dc-kitc1/virtual_envs/Fourier-Laplace/bin/activate

echo about to run python3 Server/find_configs.py $N $g $m $L

for CONFIGS in $( python3 Server/find_configs.py $N $g $m $L ); do
    echo about to run sbatch Server/run_single_thread_analysis.sh $N $g $m $L \"$T1\" \"$T2\" $CONFIGS
    
    sbatch Server/run_single_thread_analysis.sh $N $g $m $L \"$T1\" \"$T2\" $CONFIGS
done
