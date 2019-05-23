from fs_read import fs_data_collect
from select_pat_names_test_clf import select_pat_names_test_clf
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
import sys
import pandas as pd
import gbl
from gdrive import get_pat_stats
import time
from sklearn.feature_selection import VarianceThreshold
import numpy as np
from feat_select import t_frame_compute
import autosklearn.classification
import autosklearn.regression

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

# non-ocd (0-9) mild (10-20), moderate (21-30), severe (31-40) Okasha et. al. (2000)

start_time = time.time()

# get data from FreeSurfer stats
path_base = os.path.abspath('Desktop/FS_SUBJ_ALL').replace('PycharmProjects/OcdMriMl/', '')
print(path_base)
group = ['con', 'pat']

con_frame = fs_data_collect(group[0], path_base)
pat_frame = fs_data_collect(group[1], path_base)

pd.DataFrame({'con': con_frame.columns[con_frame.columns != pat_frame.columns],
              'pat': pat_frame.columns[con_frame.columns != pat_frame.columns]})

gbl.init_globals(pat_frame)
print('%d FS_FEATS READ' % len(gbl.FS_feats))

num_pats = pat_frame.shape[0]
pat_names = pat_frame.index.tolist()
num_cons = con_frame.shape[0]

pat_frame_stats = get_pat_stats()

if True in pat_frame_stats.index != pat_frame.index:
    exit("feature and target pats not same")

# remove low variance features
before = pat_frame.shape[1]
threshold = 0.01
sel = VarianceThreshold(threshold=threshold)
sel.fit_transform(pat_frame)
retained_mask = sel.get_support(indices=False)
pat_frame = pat_frame.loc[:, retained_mask]
after = pat_frame.shape[1]
print('REMOVING %d FEATS UNDER %.2f VAR: FROM %d TO %d' % (before-after, threshold, before, after))
gbl.FS_feats = pat_frame.columns.tolist()
# add demo features
gbl.demo_clin_feats = ['gender_num', 'age', 'duration', 'med']
pat_frame = pd.concat([pat_frame, pat_frame_stats.loc[:, gbl.demo_clin_feats]], axis=1, sort=False)

y_reg = pd.DataFrame({'YBOCS': pat_frame_stats.loc[:, 'YBOCS']})

clf_tgt = 'YBOCS_class3_scorerange'

y_clf = pd.DataFrame({clf_tgt: pat_frame_stats.loc[:, clf_tgt]})
num_classes = len(np.unique(y_clf))

# extract train and test set names
t_s = 0.19

pat_names_test = select_pat_names_test_clf(y_clf, clf_tgt, t_s, num_classes)
pat_names_train = [name for name in pat_names if name not in pat_names_test]
overlap = any(elem in pat_names_train for elem in pat_names_test)
print(overlap)


pat_frame_train = pat_frame.loc[pat_names_train, :]

pat_frame_train_y_reg = y_reg.loc[pat_names_train, :]
pat_frame_train_y_clf = y_clf.loc[pat_names_train, :]

pat_frame_test = pat_frame.loc[pat_names_test, :]
pat_frame_test_y_reg = y_reg.loc[pat_names_test, :]
pat_frame_test_y_clf = y_clf.loc[pat_names_test, :]

t_frame = t_frame_compute(pat_frame_train, con_frame, 0, ['thickness', 'volume'])

feats = t_frame.columns.tolist() + gbl.demo_clin_feats
print('%d features into estimator' % (len(feats)))


clf = autosklearn.classification.AutoSklearnClassifier()
clf.fit(pat_frame_train.loc[:, feats].values, pat_frame_train_y_clf.values)
clf_predictions = clf.predict(pat_frame_test.loc[:, feats].values)
clf_score = clf.score(pat_frame_test.loc[:, feats].values, pat_frame_test_y_clf.values)
#
# clf_results = pd.DataFrame(index=pat_frame_test.index.tolist(),
#                            data={'YBOCS_pred': clf_predictions, 'YBOCS_targets': pat_frame_test_y_clf.iloc[:, 0]})
# print(clf_results)
print(clf_score)

reg = autosklearn.regression.AutoSklearnRegressor()
reg.fit(pat_frame_train.loc[:, feats].values, pat_frame_train_y_reg.values)
reg_predictions = reg.predict(pat_frame_test.loc[:, feats].values)
reg_score = reg.score(pat_frame_test.loc[:, feats].values, pat_frame_test_y_reg.values)
#
# reg_results = pd.DataFrame(index=pat_frame_test.index.tolist(),
#                            data={'YBOCS_pred': reg_predictions, 'YBOCS_targets': pat_frame_test_y_reg.iloc[:, 0]})
# print(reg_results)
print(reg_score)
