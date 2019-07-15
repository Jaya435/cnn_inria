#!/bin/sh

# Grid Engine options
#
#$ -N train_titan
#$ -cwd
# -pe gpu 4 
# -l h_vmem=16G
#$ -pe gpu-titanx 2
#$ -l h_vmem=25G
#$ -l h_rt=03:00:00

# Initialise the modules framework
. /etc/profile.d/modules.sh

# Check amount of memory (in kbytes) as seen by the job
ulimit -v

# User specified commands go below here
module load anaconda/5.0.1
source activate mypytorch2


# Read a text file, containing a list of possible combinations#
input='grid_search_1.txt'
readarray myArray < "$input"
set -- ${myArray[$SGE_TASK_ID]}
# submits batch job to SGE engine
echo arch _size is "$1"

python ${HOME}/python/ConvNet.py --arch_size "$1" --lr "$2" --batch_size "$3" 
