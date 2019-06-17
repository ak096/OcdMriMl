import os
import warnings
from sklearn.exceptions import ConvergenceWarning
import sys
import pandas as pd
import global_
import time
import numpy as np
import autosklearn.classification
import autosklearn.regression
from dataset import Subs

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

# non-ocd (0-9) mild (10-20), moderate (21-30), severe (31-40) Okasha et. al. (2000)

start_time = time.time()
subs = Subs(tgt_name='YBOCS_class3_scorerange')

clf = autosklearn.classification.AutoSklearnClassifier()
clf.fit(subs.pat_frame_train.values, subs.pat_frame_train_y.values)
clf_predictions = clf.predict(subs.pat_frame_test.values)
clf_score = clf.score(subs.pat_frame_test.values, subs.pat_frame_test_y.values)
#
# clf_results = pd.DataFrame(index=pat_frame_test.index.tolist(),
#                            data={'YBOCS_pred': clf_predictions, 'YBOCS_targets': pat_frame_test_y_clf.iloc[:, 0]})
# print(clf_results)
print(clf_score)

reg = autosklearn.regression.AutoSklearnRegressor()
reg.fit(subs.pat_frame_train.values, subs.pat_frame_train_y.values)
reg_predictions = reg.predict(subs.pat_frame_test.values)
reg_score = reg.score(subs.pat_frame_test.values, subs.pat_frame_test_y.values)
#
# reg_results = pd.DataFrame(index=pat_frame_test.index.tolist(),
#                            data={'YBOCS_pred': reg_predictions, 'YBOCS_targets': pat_frame_test_y_reg.iloc[:, 0]})
# print(reg_results)
print(reg_score)
