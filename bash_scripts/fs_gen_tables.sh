#!/bin/bash

# extract features from aseg.stats and ?h.aparc.stats into table text files using \
# freesurfer *2table methods (for group ‚${SUB}={con|pat}‘ as command line input)

SUB=$1
basepathtosubjfile=~/Desktop/FS_SUBJ_ALL/${SUB}
rm -r $basepathtosubjfile/${SUB}_tables
mkdir $basepathtosubjfile/${SUB}_tables
basepathtotablesdir=$basepathtosubjfile/${SUB}_tables
export SUBJECTS_DIR=$basepathtosubjfile/${SUB}_FS_done
echo "cd to" $SUBJECTS_DIR
cd $SUBJECTS_DIR

#asegstats2table----

asegstats2table --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                --tablefile $basepathtotablesdir/${SUB}.aseg.vol.table  \
                --all-segs

asegstats2table --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                --tablefile $basepathtotablesdir/${SUB}.aseg.area.table \
                --meas Area_mm2 \
                --all-segs

#asegstats2table --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
#                --tablefile $basepathtotablesdir/${SUB}.wmparc.vol.table \
#                --stats wmparc.stats \
#                --all-segs


#aparcstats2table-----

#lh with Desikan (Default)

aparcstats2table --hemi lh \
                 --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                 --tablefile $basepathtotablesdir/${SUB}.lh.aparc.area.table \
                 --common-parcs

aparcstats2table --hemi lh \
                 --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                 --tablefile $basepathtotablesdir/${SUB}.lh.aparc.thick.table \
                 --meas thickness \
                 --common-parcs

aparcstats2table --hemi lh \
                 --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                 --tablefile $basepathtotablesdir/${SUB}.lh.aparc.vol.table \
                 --meas volume \
                 --common-parcs

#lh with Destrieux (aparc.a2009s)

aparcstats2table --hemi lh \
                 --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                 --tablefile $basepathtotablesdir/${SUB}.lh.aparc.a2009s.area.table  \
                 --parc aparc.a2009s \
                 --common-parcs

aparcstats2table --hemi lh \
                 --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                 --tablefile $basepathtotablesdir/${SUB}.lh.aparc.a2009s.thick.table \
                 --parc aparc.a2009s \
                 --meas thickness \
                 --common-parcs

aparcstats2table --hemi lh \
                 --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                 --tablefile $basepathtotablesdir/${SUB}.lh.aparc.a2009s.vol.table \
                 --parc aparc.a2009s \
                 --meas volume \
                 --common-parcs

#rh with Desikan (Default) then Destrieux (aparc.a2009s)

aparcstats2table --hemi rh \
                 --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                 --tablefile $basepathtotablesdir/${SUB}.rh.aparc.area.table \
                 --common-parcs

aparcstats2table --hemi rh \
                 --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                 --tablefile $basepathtotablesdir/${SUB}.rh.aparc.thick.table \
                 --meas thickness \
                 --common-parcs

aparcstats2table --hemi rh \
                 --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                 --tablefile $basepathtotablesdir/${SUB}.rh.aparc.vol.table \
                 --meas volume \
                 --common-parcs

#rh with Destrieux (aparc.a2009s)

aparcstats2table --hemi rh \
                 --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                 --tablefile $basepathtotablesdir/${SUB}.rh.aparc.a2009s.area.table \
                 --parc aparc.a2009s \
                 --common-parcs

aparcstats2table --hemi rh \
                 --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                 --tablefile $basepathtotablesdir/${SUB}.rh.aparc.a2009s.thick.table \
                 --parc aparc.a2009s \
                 --meas thickness \
                 --common-parcs

aparcstats2table --hemi rh \
                 --subjectsfile=$basepathtosubjfile/${SUB}_name_list.txt \
                 --tablefile $basepathtotablesdir/${SUB}.rh.aparc.a2009s.vol.table \
                 --parc aparc.a2009s \
                 --meas volume \
                 --common-parcs

cd $basepathtotablesdir
ls *.table > ${SUB}_table_filenames.txt