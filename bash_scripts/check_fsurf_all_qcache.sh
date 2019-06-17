#!/usr/bin/env bash

for d in */; do
    dir=${d%*/}
    if grep -Fxq "CMDARGS -s $dir -all -qcache" ${dir}/scripts/recon-all.done || grep -Fxq "CMDARGS -s $dir -i $dir.nii -all -qcache" ${dir}/scripts/recon-all.done
    then
        :
    else
        echo "$dir not processed -all -qcache"

    fi
    if  grep -Fxq "VERSION \$Id: recon-all,v 1.580.2.16 2017/01/18 14:11:24 zkaufman Exp $" ${dir}/scripts/recon-all.done
    then
        :
    else
        echo "$dir not processed with FS 6.0.0."
    fi
    done