#!/bin/bash
# expects argument 'con' or 'pat'
group=$1
cd ~/Desktop/FS_SUBJ_ALL/${group}
for d in */; do
    name=${d%*/}
    mkdir ~/Desktop/${group}/${name}
	cd $d
	for f in */; do
	    mv $f ~/Desktop/${group}/${name}
	done
	cd ..	
done
