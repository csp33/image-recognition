#!/bin/bash
#Skip the numbers before the name of the object
for folder in ./*; do
	if [ -d "$folder" ]; then
		echo "mv $folder ${folder:6:99}" 
	fi
done

