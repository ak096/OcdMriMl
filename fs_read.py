import pandas as pd
import subprocess
import os


def fs_data_collect(group, path_base):

    #subprocess.Popen(['/home/bran/PycharmProjects/OcdMriMl/bash_scripts/fs_gen_tables.sh %s' % group], \
    #                 shell=True, executable="/bin/bash")
    group_frame = pd.DataFrame()

    # concat fs table files into one data frame
    for table_file in os.scandir(path_base + '/' + group + '/' + group + '_tables/'):
        print("reading %s" % table_file.path)

        table = pd.read_table(table_file.path, index_col=0)

        # specify feature set (columns)
        if '.aseg.vol.' in table_file.name:
            suffix = 'volume'
        elif '.aseg.area.' in table_file.name:
            suffix = 'area'
        elif '.aparc.a2009s.' in table_file.name:
            suffix = 'aparc.a2009s'
        else:
            suffix = 'aparc'
        table.columns = [n + '**' + suffix if (n != 'BrainSegVolNotVent' and n != 'eTIV') else n for n in table.columns]

        table.columns = [s.replace('_and_', '&') for s in table.columns]

        # name the subject index
        table.index.name = 'subj'

        frames = [group_frame, table]
        group_frame = pd.concat(frames, axis=1)

    # remove duplicates (specifically BrainSegVolNotVent and eTIV features)
    group_frame = group_frame.loc[:, ~group_frame.columns.duplicated()]

    # sort by column name
    group_frame = group_frame.reindex(sorted(group_frame.columns), axis=1)

    # remove constant features (f.e. all 0's)
    # group_frame = group_frame.loc[:, group_frame.nunique() != 1]

    return group_frame
