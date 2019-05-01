import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import glob
from fs_read import fs_data_collect
from estimators import regress, classify
from find_best_model import find_best_model
from scale import scale, testSet_scale
from prediction_reporting import predict_report, write_report
from hoexter_features import hoexter_FSfeats
from boedhoe_features import boedhoe_FSfeats
from pat_names import *
from get_features import get_feats
from sklearn.model_selection import train_test_split
import time
import pickle
from pickling import *
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
import copy
import sys
import random
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.feature_selection import RFECV
from univariate import t_compute

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

# non-ocd (0-9) mild (10-20), moderate (21-30), severe (31-40) Okasha et. al. (2000)

start_time = time.time()

cv_folds = 5

# seed = 7
# np.random.seed(seed)

# get data from FreeSurfer stats
path_base = os.path.abspath('Desktop/FS_SUBJ_ALL').replace('PycharmProjects/OcdMriMl/', '')

group = ['con', 'pat']

con_frame = fs_data_collect(group[0], path_base)
pat_frame = fs_data_collect(group[1], path_base)

pd.DataFrame({'con': con_frame.columns[con_frame.columns != pat_frame.columns], 
              'pat': pat_frame.columns[con_frame.columns != pat_frame.columns]})

glob.init_globals(pat_frame)
print('%d FS_FEATS READ' % len(glob.FS_feats))

num_pats = pat_frame.shape[0]
pat_names = pat_frame.index.tolist()
num_cons = con_frame.shape[0]

num_reg = len(glob.regType_list)
num_clf = len(glob.clfType_list)


reg_scorers = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_log_error']
reg_scorers_names = ['ev', 'nmae', 'nmsle']

r_sc = 1
reg_scoring = reg_scorers[r_sc]

clf_scorers = ['balanced_accuracy', 'accuracy']
clf_scorers_names = ['bac', 'ac']

c_sc = 0
clf_scoring = clf_scorers[c_sc]


# get patient stats, labels, ybocs ...
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('ocdmriml_gdrive_client_secret.json', scope)
gsclient = gspread.authorize(creds)
pat_stats_sheet = gsclient.open("AKSHAY_pat_stats").sheet1

pat_frame_stats = pd.DataFrame(pat_stats_sheet.get_all_records())
pat_frame_stats.index = pat_frame_stats.loc[:, 'subject']
pat_frame_stats.drop(columns=['', 'subject'], inplace=True)

if True in pat_frame_stats.index != pat_frame.index:
    exit("feature and target pats not same")

# remove low variance features
before = pat_frame.shape[1]
threshold = 0.00
selector = VarianceThreshold(threshold=threshold)
selector.fit_transform(pat_frame)
retained_mask = selector.get_support(indices=False)
pat_frame = pat_frame.loc[:, retained_mask]
after = pat_frame.shape[1]
print('REMOVING %d FEATS UNDER %.2f VAR: FROM %d TO %d' % (before-after, threshold, before, after))
glob.FS_feats = pat_frame.columns.tolist()
# add demo features
demo_clin_feats = ['gender_num', 'age', 'duration', 'med']
# demo_clin_feats = []

pat_frame = pd.concat([pat_frame, pat_frame_stats.loc[:, demo_clin_feats]], axis=1, sort=False)

h_r = 'hoexter_reg'
h_c = 'hoexter_clf'
b_r = 'boedhoe_reg'
b_c = 'boedhoe_clf'
t_r = 't_reg'
t_c = 't_clf'


iteration = {'n': [0],
             'clf_targets': [#'obs_class3_scorerange',
                             #'com_class3_scorerange',
                             'YBOCS_class3_scorerange']
                             #'YBOCS_class4_scorerange']
                             #'obs_class3_equalpat',
                             #'com_class3_equalpat',
                             #'YBOCS_class3_equalpat']
             }

hoexter_feats = hoexter_FSfeats + demo_clin_feats
boedhoe_feats = boedhoe_FSfeats + demo_clin_feats

for clf_tgt in iteration['clf_targets']:

    if 'obs' in clf_tgt:
        y_reg = pd.DataFrame({'obs': pat_frame_stats.loc[:, 'obs']})
    elif 'com' in clf_tgt:
        y_reg = pd.DataFrame({'com': pat_frame_stats.loc[:, 'com']})
    else:
        y_reg = pd.DataFrame({'YBOCS': pat_frame_stats.loc[:, 'YBOCS']})

    y_clf = pd.DataFrame({clf_tgt: pat_frame_stats.loc[:, clf_tgt]})

    # extract train and test set names
    t_s = 0.15
    # pat_names_train, pat_names_test = train_test_split(pat_names,
    #                                                    test_size=t_s,
    #                                                    stratify=y_clf)
    #                                                    #random_state=random.randint(1, 101))

    pat_names_test = pat_names_test_equal_per_YBOCS_class3
    pat_names_train = [name for name in pat_names if name not in pat_names_test]
    print(len(pat_names_test))
    print(len(pat_names_train))
    pat_frame_train = pat_frame.loc[pat_names_train, :]
    pat_frame_train_norms, pat_frame_train_scalers = scale(pat_frame_train)

    pat_frame_train_y_reg = y_reg.loc[pat_names_train, :]

    pat_frame_train_y_clf = y_clf.loc[pat_names_train, :]
    # adasyn_train_clf = ADASYN(random_state=random.randint(1, 101))
    # smote_train_clf = SMOTE(random_state=random.randint(1,101))
    ros_train_clf = RandomOverSampler(random_state=random.randint(1, 101))
    pat_train_clf_ros_resamp, pat_train_y_clf_ros_resamp = ros_train_clf.fit_resample(pat_frame_train,
                                                                                      pat_frame_train_y_clf)
    pat_frame_train_clf_ros_resamp = pd.DataFrame(columns=pat_frame_train.columns,
                                                  data=pat_train_clf_ros_resamp)
    pat_frame_train_y_clf_ros_resamp = pd.DataFrame(columns=pat_frame_train_y_clf.columns,
                                                    data=pat_train_y_clf_ros_resamp)
    pat_frame_train_clf_ros_resamp_norms, pat_frame_train_clf_ros_resamp_scalers = \
        scale(pat_frame_train_clf_ros_resamp)

    # pat_test
    pat_frame_test = pat_frame.loc[pat_names_test, :]
    pat_frame_test_norms = testSet_scale(pat_frame_test, pat_frame_train_scalers)
    pat_frame_test_clf_ros_resamp_norms = testSet_scale(pat_frame_test, pat_frame_train_clf_ros_resamp_scalers)

    pat_frame_test_y_reg = y_reg.loc[pat_names_test, :]
    pat_frame_test_y_clf = y_clf.loc[pat_names_test, :]

    # con
    con_frame_norms, con_frame_scalers = scale(con_frame)

    t_reg_models_all = []
    t_clf_models_all = []

    glob.t_frame_perNorm_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

    # expert-picked-feature-based models for regression and classification
    hoexter_reg_models_all = []
    hoexter_clf_models_all = []
    boedhoe_reg_models_all = []
    boedhoe_clf_models_all = []

    for n in iteration['n']: # standard, minmax, (robust-quantile-based) normalizations of input data
        t0 = time.time()
        norm = glob.normType_list[n]

        # print("HOEXTER Regression with norm " + norm)

        # hoexter_reg_models_all += regress(pat_frame_train_norms[n][hoexter_feats],
        #                                   pat_frame_train_y_reg, cv_folds, reg_scoring, n, h_r,
        #                                   len(hoexter_feats))
        #
        # hoexter_clf_models_all += classify(pat_frame_train_norms[n][hoexter_feats],
        #                                    pat_frame_train_y_clf, cv_folds, clf_scoring, n, h_c,
        #                                    len(hoexter_feats))
        # # print("BOEDHOE Regression with norm " + norm)
        #
        # boedhoe_reg_models_all += regress(pat_frame_train_norms[n][boedhoe_feats],
        #                                   pat_frame_train_y_reg, cv_folds, reg_scoring, n, b_r,
        #                                   len(boedhoe_feats))
        #
        # boedhoe_clf_models_all += classify(pat_frame_train_norms[n][boedhoe_feats],
        #                                    pat_frame_train_y_clf, cv_folds, clf_scoring, n, b_c,
        #                                    len(boedhoe_feats))

        print("HOEXTER and BOEDHOE EST W/ NORM %s TOOK %.2f SEC" % (norm, time.time()-t0))

        # compute t_feats
        t_frame = t_compute(pat_frame_train, con_frame, n)

        t_feats_num_total = t_frame.shape[1]
        t_feats_demo_num_total = t_feats_num_total + len(demo_clin_feats)

        t_feats_list = t_frame.columns.tolist()
        print("FINISHED COMPUTING %d T VALUES" % t_feats_num_total)

        print(pd.DataFrame(data=t_feats_list))

        for idx2, t_feat in enumerate(t_feats_list):
            t0 = time.time()
            t_feats_num = idx2+1
            t_feats = t_feats_list[0:t_feats_num]

            t_reg_models_all += regress(pat_frame_train_norms[n][t_feats + demo_clin_feats],
                                        pat_frame_train_y_reg, cv_folds,
                                        reg_scoring, n, t_r, t_feats_demo_num_total)

            t_clf_models_all += classify(pat_frame_train_clf_ros_resamp_norms[n][t_feats + demo_clin_feats],
                                         pat_frame_train_y_clf_ros_resamp, cv_folds,
                                         clf_scoring, n, t_c, t_feats_demo_num_total)

            print("%d / %d T FEATS EST W/ NORM %s TOOK %.2f SEC" % (t_feats_num,
                                                                    t_feats_num_total,
                                                                    norm,
                                                                    time.time() - t0))
            # end for t_feats
        # end for n
    # end for clf_targets

    # find best trained models and prediction results
    best_models_results = {}
    models_all = {h_r: hoexter_reg_models_all, h_r: hoexter_clf_models_all,
                  b_r: boedhoe_reg_models_all, b_c: boedhoe_clf_models_all,
                  t_r: t_reg_models_all, t_c: t_clf_models_all}

    for key, value in models_all.items():
        if value:
            bm, bm5 = find_best_model(key, value, reg_scoring)
            est_type = bm['est_type']
            pat_frame_test = pat_frame_test_norms[bm['normIdx_train']]
            ft = get_feats(key, bm) + demo_clin_feats
            if est_type in glob.regType_list:
                ec = 'reg'
                pr = predict_report(key, bm, pat_frame_test, ft, pat_frame_test_y_reg, ec)
            elif est_type in glob.clfType_list:
                ec = 'clf'
                pr = predict_report(key, bm, pat_frame_test, ft, pat_frame_test_y_clf, ec)

            best_models_results[key] = {'features': ft, 'est_class': ec, 'best_model': bm, 'pred_results': pr,
                                        'bm5': bm5}

    # best_models_results[key] = {'features': [list],
    #                             'est_class': 'reg'||'clf',
    #                             'best_model': {'GridObject': , 'est_type': , 'normIdx_train': , 'num_feats': },
    #                             'pred_results': prediction_frame
    #                             'bm5': top 5 best scoring models for confidence interval comparison
    #                            }

    # SAVE RESULTS
    print('SAVING RESULTS')
    # str(t_s) + \
    exp_description = '**handTest_RegTrainRest_ClfTrainROS' + '_' + norm + '_' + reg_scorers_names[r_sc] + '_' \
                      + clf_scorers_names[c_sc] + '_' + 'cvFolds' + str(cv_folds) + \
                      '**t_allRegTrain_TorP'

    try:
        os.mkdir(clf_tgt + exp_description)
    except FileExistsError:
        pass

    bmr = open(clf_tgt + exp_description + '/' + clf_tgt + exp_description + '**bmr.pkl', 'wb')
    pickle.dump(best_models_results, bmr, -1)
    bmr.close()

    # write prediction results to excel
    xlsx_name = clf_tgt + exp_description + '/' + clf_tgt + exp_description + '**results**' + \
                'tclf:' + format(round(best_models_results[t_c]['pred_results'].iloc[-1, 0], 2))+'_'+\
                'treg:' + format(round(best_models_results[t_r]['pred_results'].iloc[-2, 0], 2))+'.xlsx'

    writer = pd.ExcelWriter(xlsx_name)
    write_report(writer, best_models_results, pat_frame_test_y_clf, pat_frame_test_y_reg)
    for idx3, tfpNl in enumerate(glob.t_frame_perNorm_list):
        tfpNl.to_excel(writer, 't_frame_' + glob.normType_list[idx3])

    writer.save()
    print(xlsx_name)

    # end for clr_tgt
print("TOTAL TIME %.2f" % (time.time()-start_time))
