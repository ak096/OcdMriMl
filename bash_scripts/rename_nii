#!/bin/bash

#rename *.nii file to patient_name.nii (from directory name)
for d in */; do
	name=${d%/*}
	cd $d
	if [ -e *.nii ]
	then 
		echo renaming ${./*.nii} to ${${name}.nii} 
		# mv ./*.nii "${name}.nii"
		# cp "${name}.nii" /home/bran/Desktop/FtoDo
	fi 
	cd ..
done
