#!/bin/sh

# Grid Engine options
# 2 compute cores, total 24 GB memory. Runtime limit of 1 hour.
#$ -N test_model
#$ -cwd
#$ -pe sharedmem 4
#$ -l h_vmem=16G
#$ -l h_rt=00:10:00

# Initialise the modules framework
. /etc/profile.d/modules.sh

# Check amount of memory (in kbytes) as seen by the job
ulimit -v

# User specified commands go below here
module load anaconda/5.0.1
source activate mypytorch 
# Run the program
./test_model.py 

