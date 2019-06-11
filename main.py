import numpy as np
import pandas as pd
from numpy.random import randint
import gbl
from estimators import regress, classify
from prediction_reporting import predict_report, write_report
from pat_sets import *
import time
import pickle
from pickling import *
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
import sys
import random
from feat_pool_univar import t_frame_compute, f_frame_compute, mi_frame_compute
from sklearn.feature_selection import RFECV
import xgboost as xgb
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier
from trained_models_analysis import models_to_results
from set_operations import powerset, subsequentset
from feat_selection_ml import rfe_cv
from dataset import Subs
start_time = time.time()
from collections import Counter
from copy import copy, deepcopy
from results import TargetResults
from estimators2 import train, pred
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, mean_absolute_error
from scipy.stats import sem, t
from scipy import mean


warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

gbl.init_globals()

# seed = 7
# np.random.seed(seed)

# credit: kite
def conf_interval(data):
    confidence = 0.95
    n = len(data)
    m = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return [m-h, m+h]


targets = [
           'YBOCS',
           'YBOCS_class2_scorerange',
            #'obs_class3_scorerange',
            #'com_class3_scorerange',
           'YBOCS_class3_scorerange',
           'YBOCS_class4_scorerange'
          ]

tgt_results = TargetResults()

for idx, tgt_name in enumerate(targets):
    zeit = time.time()

    if '_ROS' in tgt_name:
        subs.resample(over_sampler='ROS')
    elif '_SVMSMOTE' in tgt_name:
        subs.resample(over_sampler='SVMSMOTE')
    else :
        subs = Subs(tgt_name=tgt_name, test_size=0.15)
        if subs.imbalanced_classes:
            targets.insert(idx+1, tgt_name + '_ROS')
            targets.insert(idx+2, tgt_name + '_SVMSMOTE')

    n = 0
    norm = gbl.normType_list[n]

    tgt_results.add_target(tgt_name=tgt_name)

    # univariate feature pool computation:
    # compute t_feats
    if subs.resampled: # use .iloc
        a = subs.pat_frame_train.iloc[subs.pat_names_train_bins[0], :]
        b = subs.pat_frame_train.iloc[subs.pat_names_train_bins[subs.num_bins-1], :]
    else: # use .loc
        a = subs.pat_frame_train.loc[subs.pat_names_train_bins[0], :],
        b = subs.pat_frame_train.loc[subs.pat_names_train_bins[subs.num_bins-1], :]

    t_frame = t_frame_compute(a, b, feat_filter=[])  # ['thickness', 'volume'])
    t_feats = t_frame.columns.tolist()
    t_feats_num = len(t_feats)
    print("FINISHED COMPUTING %d T VALUES" % t_feats_num)

    # compute f_feats
    f_frame = f_frame_compute(subs.pat_frame_train, subs.pat_frame_train_y, subs.tgt_task, feat_filter=[])
    f_feats = f_frame.columns.tolist()
    f_feats_num = len(f_feats)
    print("FINISHED COMPUTING %d F VALUES" % f_feats_num)
    # compute mi_feats
    mi_frame = mi_frame_compute(subs.pat_frame_train, subs.pat_frame_train_y, subs.tgt_task, feat_filter=[])
    mi_feats = mi_frame.columns.tolist()
    mi_feats_num = len(mi_feats)
    print("FINISHED COMPUTING %d MI VALUES" % mi_feats_num)

    # feat pool
    feat_pool_all = t_feats + f_feats + mi_feats
    feat_pool_counts_frame = pd.DataFrame(Counter(feat_pool_all))
    print(feat_pool_counts_frame)

    feat_pool = list(set(feat_pool_all))

    tgt_results.targets[tgt_name].update({
                                            't_frame': t_frame.transpose,
                                            'f_frame': f_frame.transpose,
                                            'mi_frame': mi_frame.transpose,
                                            'feat_pool_counts_frame': feat_pool_counts_frame.transpose
                                          })
    print('feat_pool computation took %.2f' % time.time() - zeit)
    # ml feature selection computation
    zeit = time.time()
    l_feat_select, nl_feat_select = rfe_cv(task=subs.tgt_task, feat_pool=feat_pool, X=subs.pat_frame_train_norms[0],
                                           y=subs.pat_frame_train_y, cv_folds=subs.cv_folds)
    print('feat_selection RFECV computation took %.2f' % time.time() - zeit)
    lsvm_scoring=None
    if subs.tgt_task is 'clf':
        if subs.num_bins is 2:
            lsvm_params = {}
            lsvm_scoring = roc_auc_score(average='weighted')
            xgb_params = {'objective': 'binary:logistic', 'eval_metric':'auc'}
        elif subs.num_bins > 2:
            lsvm_params = {}
            lsvm_scoring = log_loss()
            xgb_params = {'objective': 'multi:softmax', 'num_class': subs.num_bins, 'eval_metric': 'mlogloss'}
    elif subs.tgt_task is 'reg':
        lsvm_params = {}
        lsvm_scoring = mean_absolute_error()
        xgb_params = {}

    feat_subsets = {
                    'learned': [l_feat_select, nl_feat_select],
                    'hoexter': [gbl.hoexter_feats_FS, gbl.hoexter_feats_FS],
                    'boedhoe': [gbl.boedhoe_feats_FS, gbl.boedhoe_feats_FS]
                    }
    for k, v in feat_subsets.items():
        zeit = time.time()
        l_feat_train = v[0] + gbl.demo_clin_feats
        nl_feat_train = v[1] + gbl.demo_clin_feats

        l_est5, l_val_scores = train(est_type='linear', task=subs.tgt_task, params=lsvm_params,
                                     X=subs.pat_frame_train_norms[0].loc[:, l_feat_train],
                                     y=subs.pat_frame_train_y,
                                     scoring=lsvm_scoring
                                     )

        l_pred_frames, l_pred_scores, l_perm_imp = pred(est5=l_est5, X=subs.pat_frame_test_norms[0],
                                                        y=subs.pat_frame_test_y)

        nl_est5, nl_val_scores = train(est_type='non-linear', task=subs.tgt_task, params=xgb_params,
                                       X=subs.pat_frame_train_norms[0].loc[:, nl_feat_train],
                                       y=subs.pat_frame_train_y)

        nl_pred_frames, nl_pred_scores, nl_perm_imp = pred(est5=nl_est5, X=subs.pat_frame_test_norms[0],
                                                           y=subs.pat_frame_test_y)


        tgt_results.targets[tgt_name][k].est_type['lsvm'].est5.update({
                                                                     'feat_sel': v[0],
                                                                     'pred_frames': l_pred_frames,
                                                                     'pred_scores': l_pred_scores,
                                                                     'conf_interval': conf_interval(l_pred_scores),
                                                                     'feat_imp_frames': l_perm_imp,
                                                                     'est5': l_est5,
                                                                     'train_scores': l_val_scores
                                                                      })
        tgt_results.targets[tgt_name][k].est_type['xgb'].est5.update({
                                                                     'feat_sel': v[1],
                                                                     'pred_frames': nl_pred_frames,
                                                                     'pred_scores': nl_pred_scores,
                                                                     'conf_interval': conf_interval(nl_pred_scores),
                                                                     'feat_imp_frames': nl_perm_imp,
                                                                     'est5': nl_est5,
                                                                     'train_scores': nl_val_scores
                                                                     })
        print('training and prediction computation took %.2f' % time.time() - zeit)

    # frequent item set mining




    # SAVE RESULTS
    print('SAVING RESULTS')
    # str(t_s) + \
    exp_description = '**balRandTest'+str(t_s)+'_RegTrainRest_ClfTrain' + over_samp_names[o_s] + '_' + norm + '_' \
                      + reg_scorers_names[r_sc] + '_' + clf_scorers_names[c_sc] + '_' + \
                      'cvFolds' + str(cv_folds) + \
                      '**t_allRegTrain_DesikanThickVolFeats_TorP'

    try:
        os.mkdir(tgt_name)
    except FileExistsError:
        pass

    bmr = open(tgt_name + '/' + tgt_name + exp_description + '**bmr.pkl', 'wb')
    pickle.dump(gbl.best_models_results, bmr, -1)
    bmr.close()
    try:
        t_reg_best_score = format(round(gbl.best_models_results[gbl.t_c]['pred_results'].iloc[-1, 0], 2))
    except:
        t_reg_best_score = -1
    try:
        t_clf_best_score = format(round(gbl.best_models_results[gbl.t_r]['pred_results'].iloc[-2, 0], 2))
    except:
        t_clf_best_score = -1
    # write prediction results to excel
    xlsx_name = tgt_name + '/' + tgt_name + exp_description + '**results**' + \
                'tclf:' + str(t_clf_best_score) +'_' +\
                'treg:' + str(t_reg_best_score) +'.xlsx'

    writer = pd.ExcelWriter(xlsx_name)
    write_report(writer, subs.pat_frame_test_y_clf, subs.pat_frame_test_y_reg)
    frame_name_suffix = '_non-resampled' # SMOTE, ROS, ADASYN
    gbl.t_frame_global.to_excel(writer, 't_frame' + frame_name_suffix)
    gbl.f_frame_global.to_excel(writer, 'f_frame' + frame_name_suffix)
    gbl.mi_frame_global.to_excel(writer, 'mi_frame' + frame_name_suffix)
    feat_pool_counts_frame.to_excel(writer, 'feat_pool_counts')
    writer.save()
    print('SAVED %s' % xlsx_name)

# end for tgt
t_feats_pats_cons_all = t_frame_compute(subs.pat_frame_train, subs.con_frame, []) # ['thickness', 'volume'])
writer = pd.ExcelWriter('t_frame_pats_v_cons' + frame_name_suffix)
gbl.t_frame_global.to_excel(writer)
print("TOTAL TIME %.2f" % (time.time()-start_time))
