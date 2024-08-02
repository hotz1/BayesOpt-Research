#!/bin/bash

# Define the number of jobs to submit
# NUM_JOBS=10

# Define the delay between job starts (in seconds)
# DELAY=60

# Define the Slurm job script
cd ~/scratch/BayesOpt-Research
JOB_SCRIPT="Code/EULBO/shells/run_elbo_fixed_sims.sh"

# Submit the first job
PREV_JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
echo "Submitted job 1 with Job ID: $PREV_JOB_ID"

# Loop to submit the remaining jobs with dependencies
for (( i=2; i<=NUM_JOBS; i++ ))
do
  # Submit the job with dependency on the previous job
  JOB_ID=$(sbatch --dependency=after:$PREV_JOB_ID+$DELAY "$JOB_SCRIPT" | awk '{print $4}')
  
  # Print the job ID for tracking
  echo "Submitted job $i with Job ID: $JOB_ID"
  
  # Update the previous job ID to the current one
  PREV_JOB_ID=$JOB_ID
done
