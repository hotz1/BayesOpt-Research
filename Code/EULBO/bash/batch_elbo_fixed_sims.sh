#!/bin/bash
#SBATCH --output="/home/joeyhotz/scratch/BayesOpt-Research/slurm-logs/slurm-%j.out"

# Script to run one single job 
cd ~/scratch/BayesOpt-Research
JOB_SCRIPT="Code/EULBO/bash/run_elbo_fixed_sims.sh"

# Submit the first job
PREV_JOB_ID=$(sbatch --gpus-per-node=$JOBGPUS --mem=$JOBMEM --time=$JOBTIME --export=ALL "$JOB_SCRIPT" | awk '{print $4}')
echo "Submitted job 1 with Job ID: $PREV_JOB_ID"

# Loop to submit the remaining jobs with dependencies
for (( i=2; i<=NUM_JOBS; i++ ))
do
  # Submit the job with dependency on the previous job
  JOB_ID=$(sbatch --gpus-per-node=$JOBGPUS --mem=$JOBMEM --time=$JOBTIME --export=ALL --dependency=after:$PREV_JOB_ID+$DELAY "$JOB_SCRIPT" | awk '{print $4}')
  
  # Print the job ID for tracking
  echo "Submitted job $i with Job ID: $JOB_ID"
  
  # Update the previous job ID to the current one
  PREV_JOB_ID=$JOB_ID
done
