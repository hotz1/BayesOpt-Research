#!/bin/sh
#SBATCH --output="/home/joeyhotz/scratch/BayesOpt-Research/slurm-logs/out/slurm-%j.out"
#SBATCH --error="/home/joeyhotz/scratch/BayesOpt-Research/slurm-logs/err/slurm-%j.err"

module load cuda
module load python/3.10
source ~/envs/research/bin/activate

cd ~/scratch/BayesOpt-Research
cmd="python Code/EULBO/elbo_simulate_fire.py ELBO_Fixed --N_init=${N_init} --n_actions=${n_actions} --n_simulations=${n_simulations} --n_epochs=${n_epochs}"
echo $cmd
echo ""
eval $cmd
