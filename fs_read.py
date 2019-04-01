import pandas as pd
import subprocess
import os


def fs_data_collect(group, path_base):

    # subprocess.Popen(['/fs_gen_tables %s' % group], shell=True, executable="/bin/bash")
    group_table = pd.DataFrame()

    # concat fs table files into one data frame
    for table_file in os.scandir(path_base + group + '/' + group + '_tables/'):
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
        table.columns = [n + '**' + suffix if n != 'BrainSegVolNotVent' and n != 'eTIV' else n for n in table.columns]

        table.columns = [s.replace('_and_', '&') for s in table.columns]

        # name the subject index
        table.index.name = 'subj'

        frames = [group_table, table]
        group_table = pd.concat(frames, axis=1)

    # remove duplicates (BrainSegVolNotVent and eTIV features)
    group_table = group_table.loc[:, ~group_table.columns.duplicated()]

    return group_table
