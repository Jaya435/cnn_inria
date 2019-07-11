#!/bin/sh

# Grid Engine options
#
#$ -N model_eval
#$ -cwd
#$ -pe gpu 4 
#$ -l h_vmem=16G
#$ -l h_rt=03:00:00

# Initialise the modules framework
. /etc/profile.d/modules.sh

# Check amount of memory (in kbytes) as seen by the job
ulimit -v

# User specified commands go below here
module load anaconda/5.0.1
source activate mypytorch
input='grid_search_1.txt'
readarray myArray < "$input"
set -- ${myArray[$SGE_TASK_ID]}
#job.sh $1.$SGE_TASK_ID
echo arch _size is "$1"
./model_eval_iter.py --arch_size "$1" --lr "$2" --batch_size "$3" 



