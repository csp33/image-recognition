#!/bin/bash
#Skip the numbers before the name of the object
for file in ./*; do
	mv $file ${file:6:99}
done

