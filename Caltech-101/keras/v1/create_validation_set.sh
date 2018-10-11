#!/bin/bash

#Carlos Sánchez Páez, 2018

###########################################################################
# 10 % of the images

cd train_images;

find . -type f -exec dirname {} + | uniq -c |
while read n d;do
  echo "Directory:$d Files:$n Moving first:$(($n / 10))";
  mkdir -p ../validation_images${d:1};find $d -type f | head -n $(($n / 10)) |
  while read file;do
    mv $file ../validation_images${d:1}/;
  done;
done
