#!/bin/bash

N=$1
G=$2
M=$3
L=$4
T1=\'$5\'
T2=\'$6\'
XMAX=$7
DIMS=$8
SIZE=$9
OR=${10}

if (( $# != 10 )); then
    echo '' 1>&2
    echo "usage: `basename $0` <N> <g> <m> <L> <T1> <T2> <x_max> <dims> <size> <OR>" 1>&2
    exit 1
fi

cd /rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-meas/ben/Fourier_Laplace

echo ./create-run-dir.sh $G $N $L $M $OR
./create-run-dir.sh $G $N $L $M $OR

echo ./create-local-templates.sh $G $N $L $M $OR
./create-local-templates.sh $G $N $L $M $OR

cd /rds/project/dirac_vol4/rds-dirac-dp099/ben/Fourier-Laplace

source /home/dc-kitc1/virtual_envs/Fourier-Laplace/bin/activate

# First things first we need to detect all of the relevent configs in this analysis
echo python3 Server/find_configs.py $N $G $M $L $SIZE
CONFIGS=$( python3 Server/find_configs.py $N $G $M $L $SIZE )
IFS=' ' read -a CONFIGS <<< "$CONFIGS"
LEN_CONFIGS=${#CONFIGS[@]}
echo $LEN_CONFIGS

# Before running the full analysis we need to calcualte the full one-point function
# myargs="$N $G $M $L \'$T1\'"
# echo $myargs | xargs python3 Server/calculate_onept.py

# myargs="$N $G $M $L \'$T2\'"
# echo $myargs | xargs python3 Server/calculate_onept.py

# Edit the array script to use the number of configs
cp Server/analysis_array.sh Server/temp.sh
sed "s/--array=/--array=0-$(( $LEN_CONFIGS - 1 ))/g" Server/temp.sh >> Server/temp2.sh
mv Server/temp2.sh Server/temp.sh

# sed "s/--array=/--array=66-66/g" Server/temp.sh >> Server/temp2.sh
# mv Server/temp2.sh Server/temp.sh

# sed "s/--output=/--output=Server/log/Fourier_Laplace_N${N}_g{$G}_m{$M}_L{$L}-%A-%a.out" Server/temp.sh >> Server/temp2.sh
# mv Server/temp2.sh Server/temp.sh

# sed "s/--error=/--error=Server/log/Fourier_Laplace_N${N}_g{$G}_m{$M}_L{$L}-%A-%a.out" Server/temp.sh >> Server/temp2.sh
# mv Server/temp2.sh Server/temp.sh

# Change the parameters to match what we have
sed "s/^N=/N=$N/" Server/temp.sh >> Server/temp2.sh
mv Server/temp2.sh Server/temp.sh

sed "s/^G=/G=$G/" Server/temp.sh >> Server/temp2.sh
mv Server/temp2.sh Server/temp.sh

sed "s/^M=/M=$M/" Server/temp.sh >> Server/temp2.sh
mv Server/temp2.sh Server/temp.sh

sed "s/^L=/L=$L/" Server/temp.sh >> Server/temp2.sh
mv Server/temp2.sh Server/temp.sh

sed "s/^T1=/T1=$T1/" Server/temp.sh >> Server/temp2.sh
mv Server/temp2.sh Server/temp.sh

sed "s/^T2=/T2=$T2/" Server/temp.sh >> Server/temp2.sh
mv Server/temp2.sh Server/temp.sh

sed "s/^SIZE=/SIZE=$SIZE/" Server/temp.sh >> Server/temp2.sh
mv Server/temp2.sh Server/temp.sh

sed "s/^XMAX=/XMAX=$XMAX/" Server/temp.sh >> Server/temp2.sh
mv Server/temp2.sh Server/temp.sh

sed "s/^DIMS=/DIMS=$DIMS/" Server/temp.sh >> Server/temp2.sh
mv Server/temp2.sh Server/temp.sh

# Submit the script
sbatch Server/temp.sh
# rm Server/temp.sh
# bash Server/temp.sh
