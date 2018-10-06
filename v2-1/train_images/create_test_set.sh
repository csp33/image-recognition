#!/bin/bash

#Carlos Sánchez Páez, 2018

###########################################################################

#Initialize variables

test_files=10


#Iterate through the folder

for folder in ./*; do
  cont=0
  mkdir ../test_images/${folder:2:99}
  for file in $folder/*.jpg; do
    if [[ $cont -ne $test_files ]]; then
      let "cont++"
      mv $file ../test_images/${folder:2:99}/
    fi
  done
done
