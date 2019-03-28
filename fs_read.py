import pandas as pd
import subprocess


def fs_data_collect(group, path_base):

	# subprocess.Popen(['/home/bran/Desktop/bash_scripts/fs_gen_tables %s' % group], shell=True, executable="/bin/bash")
	group_table	= pd.DataFrame()

	f = open(path_base + group + '_tables/' + group + '_table_filenames.txt')
	for line in f:
		filename = line.rstrip('\n')
		table = pd.read_table(path_base + group + '_tables/' + filename, index_col=0)
		# rename columns
		measure = table.index.name
		table.index.name = 'subj'
		table.columns = [n + '**' + measure for n in table.columns]
		table.columns = [s.replace('_and_', '&') for s in table.columns]

		frames = [group_table, table]
		group_table = pd.concat(frames, axis=1)
	f.close()
	#print(group_table.columns.size)
	group_table = group_table.groupby(lambda s: s, axis=1).sum()
	group_table.sort_index(axis=1, inplace=True)

	#remove duplicate BrainSegVolNotVent and eTIV features
	#print(group_table.columns.size)
	#Bcount = 0
	#ecount = 0
	for x in group_table.columns.tolist():
		if (x != "BrainSegVolNotVent**Measure:volume") and (x.split('**')[0] == "BrainSegVolNotVent"):
			group_table.drop(columns=x, inplace=True)
			#Bcount= Bcount+1
		if x.split("**")[0] == "eTIV":
			group_table.rename({x: "eTIV"}, axis='columns', inplace=True)
			#ecount= ecount+1
	#print(Bcount)
	#print(ecount)
	#print(group_table.columns.size)
	group_table = group_table.loc[:, ~group_table.columns.duplicated()]
	#print(group_table.columns.size)
	#todo remove columns with only zeros

	return group_table