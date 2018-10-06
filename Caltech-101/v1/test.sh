#!/bin/bash

#Script to test a deep neural net.
#Carlos Sánchez Páez, 2018

###########################################################################

# Test if the program was launched correctly
if [ "$#" -ne 2 ]; then
    echo "Use: $0 <test folder> <kind of images>"
    exit 1
fi

#Initialize variables

correct_predictions=0
incorrect_predictions=0
total_files=$(ls -l $1 | grep jpg | wc -l) #The number of files to be processed

#Iterate through the folder

for filename in $1/*.jpg; do
    result=$(python recognise.py $filename | tail -1); #Store the result
    if [ "$result" == "$2" ]; then  #Increment the corresponding variable
      let "correct_predictions++"
    else
      let "incorrect_predictions++"
    fi
    let "current=correct_predictions+incorrect_predictions"
    echo "Correct predictions: $correct_predictions Incorrect predictions: $incorrect_predictions Progress: $current/$total_files"
done

#Export the data to a file

d=`date +%d_%m_%y`
output="results_${d}_$2.txt"
echo "Exporting final results to $output"
echo "Correct predictions: $correct_predictions Incorrect predictions: $incorrect_predictions Number of predictions: $total_files" > ./stadistics/$output
