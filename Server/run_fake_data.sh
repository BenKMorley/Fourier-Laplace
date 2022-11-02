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

#SBATCH --output=Server/log/Fake_data-%A.out
#SBATCH --error=Server/log/Fake_data-%A.err
#SBATCH -A dirac-dp099-cpu          # Which project to charge
#SBATCH -p icelake
#SBATCH --array=0-999
#SBATCH --requeue

#SBATCH --job-name=Fake_data   # Job name
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=bkm1n18@soton.ac.uk    # Where to send mail	

#! Don't put any #SBATCH directives below this line

#! Modify the environment seen by the application. For this example we need the default modules.
. /etc/profile.d/modules.sh                # This line enables the module command
module purge                               # Removes all modules still loaded
module load rhel7/default-peta4            # REQUIRED - loads the basic environment

#! The variable $SLURM_ARRAY_TASK_ID contains the array index for each job.
#! In this example, each job will be passed its index, so each output file will contain a different value
source /home/dc-kitc1/virtual_envs/Fourier-Laplace/bin/activate

python3 Server/run_make_fake_data.py $SLURM_ARRAY_TASK_ID
