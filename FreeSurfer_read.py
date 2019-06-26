import pandas as pd
import subprocess
import os


def FreeSurfer_data_collect(group, path_base):

    #subprocess.Popen(['/home/bran/PycharmProjects/OcdMriMl/bash_scripts/fs_gen_tables.sh %s' % group], \
    #                 shell=True, executable="/bin/bash")
    group_frame = pd.DataFrame()
    repeats = ['BrainSegVolNotVent', 'eTIV', 'Right-Accumbens-area', 'Left-Accumbens-area']
    # concat fs table files into one data frame
    for table_file in os.scandir(path_base + '/' + group + '/' + group + '_tables/'):
        #print("reading %s" % table_file.path)

        table = pd.read_table(table_file.path, index_col=0)

        # feature renaming (sub-parameters: 'thickness', 'area', 'volume')

        # two main cases of features : aseg/wmparc for which sub-parameters lacks in name,
        #                              aparc(Desikan Atlas)/aparc.a2009s(Destreaux Atlas) which must be distinguished
        if any(s in table_file.name for s in ['.aseg.vol.', '.wmparc.vol.']):
            suffix = '_volume**SubCort'
        elif '.aseg.area.' in table_file.name:
            suffix = '_area**SubCort'
        # 'thickness' sub-parameter NOT available in sub-cortical (aseg) measurements
        elif '.aparc.a2009s.' in table_file.name:
            suffix = '**Dest.09s'
        else:
            suffix = '**Desi'
        # '_area', '_volume', '_thickness' already suffixed in aparc(.a2009s.) feature names

        # suffixes have ** to make unique so no chance of feat name having it when filtering atlases
        table.columns = [n + suffix if n not in repeats else n for n in table.columns]

        table.columns = [s.replace('_and_', '&') for s in table.columns]

        # name the subject index
        table.index.name = 'subj'

        frames = [group_frame, table]
        group_frame = pd.concat(frames, axis=1)

    # remove duplicates (specifically BrainSegVolNotVent and eTIV features)
    group_frame = group_frame.loc[:, ~group_frame.columns.duplicated()]
    # add back suffix to singleton repeats
    for r in repeats:
        group_frame = group_frame.rename(columns={r: r+'**SubCort'})
    # sort by column name
    group_frame = group_frame.reindex(sorted(group_frame.columns), axis=1)

    # remove constant features (f.e. all 0's)
    # group_frame = group_frame.loc[:, group_frame.nunique() != 1]
    writer1 = pd.ExcelWriter('FreeSurfer_features.xlsx')
    group_frame.to_excel(writer1, 'FreeSurfer_feats')
    writer1.save()
    return group_frame
