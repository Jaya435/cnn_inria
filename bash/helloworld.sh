#!/bin/sh
###########################################
#                                         #
# Submit a simple "Hello World" style job #
#                                         #
###########################################
 
# Grid Engine options can go here (always start with #$ )
#$ -N helloworld
#$ -cwd
#$ -l h_vmem=2G
#$ -l h_rt=00:02:00
 
# Run the program below here (This is a simple shell script)
echo '======================================================================'
echo 'Hello World!'
echo "This job ran on $HOSTNAME"
echo 'The current date and time is: ' $(date)
echo '======================================================================'
