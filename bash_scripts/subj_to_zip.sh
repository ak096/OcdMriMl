#!/bin/bash

#${SUB}={con|pat}‘ as command line input

SUB=$1
basepathtosubjfile=/home/bran/Desktop/FS_SUBJ_ALL
cd $basepathtosubjfile/${SUB}
mkdir $basepathtosubjfile/${SUB}_zip
for d in */; do
    name=${d%/*} 
	zip -r $basepathtosubjfile/${SUB}_zip/$name.zip $name 
done

