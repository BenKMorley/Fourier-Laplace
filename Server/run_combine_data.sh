#!/bin/bash
#! This line is a comment
#! Make sure you only have comments and #SBATCH directives between here and the end of the #SBATCH directives, or things will break
#! Name of the job:
#SBATCH -J test_dmtcp
#! Account name for group, use SL2 for paying queue:
#SBATCH -A MYPROJECT-CPU
#! Output filename:
#! %A means slurm job ID and %a means array index
#SBATCH --output=test_dmtcp_%A_%a.out
#! Errors filename:
#SBATCH --error=test_dmtcp_%A_%a.err

#! Number of nodes to be allocated for the job (for single core jobs always leave this at 1)
#SBATCH --nodes=1
#! Number of tasks. By default SLURM assumes 1 task per node and 1 CPU per task. (for single core jobs always leave this at 1)
#SBATCH --ntasks=1
#! How many many cores will be allocated per task? (for single core jobs always leave this at 1)
#SBATCH --cpus-per-task=1
#! Estimated runtime: hh:mm:ss (job is force-stopped after if exceeded):
#SBATCH --time=20:00:00   
#! Estimated maximum memory needed (job is force-stopped if exceeded):
#! RAM is allocated in ~5980mb blocks, you are charged per block used,
#! and unused fractions of blocks will not be usable by others.
#! Submit a job array with index values between 0 and 31
#! NOTE: This must be a range, not a single number (i.e. specifying '32' here would only run one job, with index 32)
#SBATCH --array=0-0

#SBATCH --output=Server/log/Combine_data-%A-%a.out
#SBATCH --error=Server/log/Combine_data-%A-%a.err
#SBATCH -A dirac-dp099-cpu          # Which project to charge
#SBATCH -p icelake

#SBATCH --job-name=Combine_data   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bkm1n18@soton.ac.uk    # Where to send mail	

#! Don't put any #SBATCH directives below this line

#! Modify the environment seen by the application. For this example we need the default modules.
. /etc/profile.d/modules.sh                # This line enables the module command
module purge                               # Removes all modules still loaded
module load rhel7/default-peta4            # REQUIRED - loads the basic environment

#! The variable $SLURM_ARRAY_TASK_ID contains the array index for each job.
#! In this example, each job will be passed its index, so each output file will contain a different value
echo "This is job" $SLURM_ARRAY_TASK_ID
ID=$SLURM_ARRAY_TASK_ID

N=2
G=0.2
M=-0.062
L=256
T1='(0,0)'
T2='(1,1)'

XMAXS=( 1 2 3 4 8 16 32 64 128 256 )
LEN_XMAX=${#XMAXS[@]}
X_INDEX=$(( ID%LEN_XMAX ))
XMAX=${XMAXS[$X_INDEX]}

source /home/dc-kitc1/virtual_envs/Fourier-Laplace/bin/activate

python3 Server/combine_data.py $N $G $M $L \'$T1\' \'$T2\' $XMAX
