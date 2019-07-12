#!/bin/sh
###########################################
#                                         #
# This job predicts and plots where the   #
# buildings are on an RGB image.          #
#                                         #
###########################################

# Grid Engine Options
#$ -N Accuracy
#$ -cwd
#$ -l h_rt=01:00:00
#$ -pe sharedmem 16
#$ -l h_vmem=8G

# Initialise the modules framework
. /etc/profile.d/modules.sh

# Check amount of memory (in kbytes) as seen by the job
ulimit -v

# User specified commands go below here
module load anaconda/5.0.1
source activate mypytorch
# Run the program
python ${HOME}/python/accuracy.py
