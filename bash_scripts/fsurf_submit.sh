#!/bin/bash

# submit jobs to fsurf (open science grid supercomputer)

for f in *.zip; do
    ~/Desktop/Skripte/fsurf submit --subject=${f%.*} --input=$f --defaced --deidentified --version 6.0.0 --dualcore --freesurfer-options='-all -qcache' 
done
