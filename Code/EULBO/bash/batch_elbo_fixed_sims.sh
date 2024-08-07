#!/bin/bash
#SBATCH --output="/home/joeyhotz/scratch/BayesOpt-Research/slurm-logs/slurm-%j.out"

module load cuda
module load python/3.10
source ~/envs/research/bin/activate

# Script to run the desired job 
cd ~/scratch/BayesOpt-Research
cmd="python Code/EULBO/elbo_simulate_fire.py ELBO_Fixed --N_init=${N_init} --n_actions=${n_actions} --n_simulations=${n_simulations} --n_epochs=${n_epochs}"

# Submit the first job
PREV_JOB_ID=$(sbatch --gpus-per-node=1 --cpus-per-task=1 --mem=$JOBMEM --time=$JOBTIME $cmd | awk '{print $4}')
echo "Submitted job 1 with Job ID: $PREV_JOB_ID"

# Loop to submit the remaining jobs with dependencies
for (( i=2; i<=NUM_JOBS; i++ ))
do
  # Submit the job with dependency on the previous job
  JOB_ID=$(sbatch --gpus-per-node=1 --cpus-per-task=1 --mem=$JOBMEM --time=$JOBTIME --dependency=after:$PREV_JOB_ID+$DELAY $cmd | awk '{print $4}')
  # Print the job ID for tracking
  echo "Submitted job $i with Job ID: $JOB_ID"
  
  # Update the previous job ID to the current one
  PREV_JOB_ID=$JOB_ID
done
