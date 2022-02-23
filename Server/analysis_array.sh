#!/bin/bash
#! This line is a comment
#! Make sure you only have comments and #SBATCH directives between here and the end of the #SBATCH directives, or things will break
#! Name of the job:
#SBATCH -J test_dmtcp

#! Number of nodes to be allocated for the job (for single core jobs always leave this at 1)
#SBATCH --nodes=1
#! Number of tasks. By default SLURM assumes 1 task per node and 1 CPU per task. (for single core jobs always leave this at 1)
#SBATCH --ntasks=1
#! How many many cores will be allocated per task? (for single core jobs always leave this at 1)
#SBATCH --cpus-per-task=56
#! Estimated runtime: hh:mm:ss (job is force-stopped after if exceeded):
#SBATCH --time=36:00:00

#SBATCH --array=
#SBATCH --output=Server/log/Fourier_Laplace-%A-%a.out
#SBATCH --error=Server/log/Fourier_Laplace-%A-%a.err
#SBATCH -A dirac-dp099-cpu          # Which project to charge
#SBATCH -p cclake

#SBATCH --job-name=Fourier_Laplace   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bkm1n18@soton.ac.uk    # Where to send mail
#SBATCH --requeue                   ### On failure, requeue for another try

#! Modify the environment seen by the application. For this example we need the default modules.
. /etc/profile.d/modules.sh                # This line enables the module command
module purge                               # Removes all modules still loaded
module load rhel7/default-peta4            # REQUIRED - loads the basic environment

#! The variable $SLURM_ARRAY_TASK_ID contains the array index for each job.
#! In this example, each job will be passed its index, so each output file will contain a different value
echo "This is job" $SLURM_ARRAY_TASK_ID
ID=$SLURM_ARRAY_TASK_ID

N=
G=
M=
L=
SIZE=
T1_1=
T1_2=
T2_1=
T2_2=
XMAX=
DIMS=

source /home/dc-kitc1/virtual_envs/Fourier-Laplace/bin/activate

echo python3 Server/find_configs.py $N $G $M $L $SIZE -ID=$ID
CONFIGS=$( python3 Server/find_configs.py $N $G $M $L $SIZE -ID=$ID )
IFS=' ' read -a CONFIGS <<< "$CONFIGS"
LEN_CONFIGS=${#CONFIGS[@]}
echo ${CONFIGS[@]}
echo $LEN_CONFIGS

# Run the Hadrons Measurement Code
cd /rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-meas/ben/Fourier_Laplace

# Assume that the configs are evenly 
START=${CONFIGS[0]}
END=${CONFIGS[-1]}
NEXT=${CONFIGS[1]}
STEP=$(( NEXT - START ))
NCONF=$(( ( END - START )/STEP + 1 ))
echo $START
echo $NEXT
echo $END
echo $STEP

set -e

function replace () {
    local FILE=$1
    local FIELD=$2
    local VAL=$3
    local TMP=`mktemp`
    sed "s#@${FIELD}@#${VAL}#g" ${FILE} > ${TMP}
    mv ${TMP} ${FILE}
    rm -f ${TMP}
}

CWD=`pwd -P`

RUNDIR="g$G/su$N/L$L/m2$M"
if [ -d ${RUNDIR} ]; then
    cd ${RUNDIR}
else
    echo "error: no run directory ${RUNDIR}" 1>&2
    exit 1
fi

if [ ! -e .template.xml ] || [ ! -e .template.sh ]; then
    echo "error: no local templates, cf. create-local-templates.sh" 1>&2
    exit 1
fi

echo "* ${START}"
echo "-- generating parameters and scripts..."
mkdir -p par
INPUTPART=`readlink .template.xml`
INPUT="par/${INPUTPART//xml/$START.xml}"
mkdir -p par
cp ${INPUTPART} ${INPUT}
echo ${INPUT}
replace ${INPUT} 'xi'         '0.'
replace ${INPUT} 'start'      ${START}
replace ${INPUT} 'step'       ${STEP}
replace ${INPUT} 'end'        $(( END + STEP ))
# replace ${INPUT} 'configstem' `find config/ -name "*.${conf}" | sed "s/.${conf}//g"`
echo "-- Running Hadrons ..."

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=1
numtasks=1
mpi_tasks_per_node=1

#! Environment
export PATH=/rds/project/dirac_vol4/rds-dirac-dp099/env/:${PATH}

. /etc/profile.d/modules.sh # Leave this line (enables the module command)
source mod-intel18.sh

export KMP_AFFINITY=compact
export I_MPI_PIN=1
export OMP_NUM_THREADS=56
export COMMS_THREADS=8
export I_MPI_THREAD_SPLIT=1
export I_MPI_THREAD_RUNTIME=openmp
export I_MPI_THREAD_MAX=${COMMS_THREADS}
export PSM2_MULTI_EP=1
export I_MPI_EXTRA_FILESYSTEM=on
export I_MPI_EXTRA_FILESYSTEM_LIST=lustre
export I_MPI_LUSTRE_STRIPE_AWARE=on

#! Main command
np=$[${numnodes}*${mpi_tasks_per_node}]
application="/rds/project/dirac_vol4/rds-dirac-dp099/ben/Hadrons-211117/Hadrons/build/utilities/HadronsXmlRun"
options="${INPUT} --grid ${L}.${L}.${L} --mpi 1.1.1 --debug-signals --decomposition --threads ${OMP_NUM_THREADS} --shm 2048"
CMD="mpirun -ppn $mpi_tasks_per_node -np $np $application $options"

echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Array ID: $SLURM_ARRAY_TASK_ID\n"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

echo $SLURM_JOB_NODELIST

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
	cat $NODEFILE
        mkdir -p machine
        cat $NODEFILE | uniq > machine/$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine/$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nCheck links:\n==================\n"
ldd $application

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD

echo -e "\nEND ==================\n"


# Now run the python analysis on this config
cd /rds/project/dirac_vol4/rds-dirac-dp099/ben/Fourier-Laplace
NAME=Server/run_3D_analysis.py

echo python3 Server/find_configs.py $N $G $M $L $SIZE
CONFIG_ARRAY=$( python3 Server/find_configs.py $N $G $M $L $SIZE )
IFS=' ' read -a CONFIG_ARRAY <<< "$CONFIG_ARRAY"
CONFIGS=${CONFIG_ARRAY[$ID]}

echo "python3 $NAME $N $G $M $L $T1_1 $T1_2 $T2_1 $T2_2 '$CONFIGS' -x_max=$XMAX -dims=$DIMS"

python3 $NAME $N $G $M $L $T1_1 $T1_2 $T2_1 $T2_2 \'$CONFIGS\' -x_max=$XMAX -dims=$DIMS

for conf in $(seq ${START} ${STEP} ${END})
do
    # Delete the correlator
    cd /rds/project/dirac_vol4/rds-dirac-dp099/cosmhol-meas/ben/Fourier_Laplace
    cd $RUNDIR

    rm FL/cosmhol-su${N}_L${L}_g${G}_m2${M}-FL.${conf}.h5
done
