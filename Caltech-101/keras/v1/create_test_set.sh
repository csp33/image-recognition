#!/bin/bash

#Carlos Sánchez Páez, 2018

###########################################################################
# 15 % of the images

cd train_images;
find . -type f -exec dirname {} + | uniq -c |
while read n d;do
  echo "Directory:$d Files:$n Moving first:$(($n / 15))";
  mkdir -p ../test_images${d:1};find $d -type f | head -n $(($n / 15)) |
  while read file;do
    mv $file ../test_images${d:1}/;
  done;
done
