#!/bin/bash

#Script to test a deep neural net.
#Carlos Sánchez Páez, 2018

###########################################################################


#Initialize variables

correct_predictions=0
incorrect_predictions=0

total_files=$(ls -lR ./test_images/ | grep jpg | wc -l) #The number of files to be processed


for folder in ./test_images/*; do
  if [[ -d $folder ]]; then
    echo "Now testing ${folder:14:99} images..."
    for filename in $folder/*.jpg; do
        result=$(python recognise.py $filename | tail -1); #Store the result
        if [ "$result" == "${folder:14:99}" ]; then  #Increment the corresponding variable
          let "correct_predictions++"
        else
          let "incorrect_predictions++"
        fi
        let "current=correct_predictions+incorrect_predictions"
        echo "Correct predictions: $correct_predictions Incorrect predictions: $incorrect_predictions Progress: $current/$total_files"
    done
  fi
done

d=`date +%d_%m_%y`
output="results_${d}.txt"
echo "Exporting final results to $output"
echo "Correct predictions: $correct_predictions Incorrect predictions: $incorrect_predictions Number of predictions: $total_files" > ./stadistics/$output
