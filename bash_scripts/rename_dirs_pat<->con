#!/bin/bash

# expects argument 'con' or 'pat'

group=$1

for d in */; do
	dir=${d%*/}
	# echo $dir
	sub=${dir:4}
	name="${group}_$sub"
	echo "renaming $name"
	mv $dir $name
done
