#!/bin/bash

input='grid_search.txt'

readarray myarray < "$input"
for num in 1 2 3;
do
    set -- ${myarray["$num"]}
    echo $3
done
echo ${myarray[1]}
#set -- ${myarray[0]}
#echo $2
