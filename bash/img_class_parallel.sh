##!/bin/sh

# Grid Engine options
#
#$ -N GPU-K80
#$ -cwd
#$ -pe gpu 4 
#$ -l h_vmem=16G
#$ -l h_rt=02:00:00

# Initialise the modules framework
. /etc/profile.d/modules.sh

# Check amount of memory (in kbytes) as seen by the job
ulimit -v

# User specified commands go below here
module load anaconda/5.0.1
source activate mypytorch
# Run the program
./model_eval.py

