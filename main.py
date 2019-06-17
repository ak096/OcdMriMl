import numpy as np
import pandas as pd
from numpy.random import randint
import gbl
from estimators_old import regress, classify
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
from feat_selection_ml import rfe_cv, freq_item_sets
from dataset import Subs
from collections import Counter
from copy import copy, deepcopy
from results import FeatSetResults
from train_predict import train, pred
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, mean_absolute_error
from scipy.stats import sem, t
from scipy import mean
import datetime

from scorers_ import RegScorer, ClfScorer

start_time = time.time()
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

pd.options.mode.use_inf_as_na = True

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
           'YBOCS_class2',
            #'obs_class3',
            #'com_class3',
           'YBOCS_class3',
           'YBOCS_class4'
          ]

all_tgt_results = {}

lrn_feat_sets = {gbl.linear_: [], gbl.non_linear_: []}

clf_scorer = ClfScorer()
reg_scorer = RegScorer()

feat_sets_count = 0

for idx, tgt_name in enumerate(targets):

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

    all_tgt_results[tgt_name] = {
                                't_frame': None,
                                'f_frame': None,
                                'mi_frame': None,
                                'feat_count_frame': None,
                                gbl.linear_: {},
                                gbl.non_linear_: {}
                                }


    # univariate feature pool computation:
    print('%s : starting feat_pool computation'% (tgt_name))
    zeit = time.time()
    # compute t_feats
    if subs.resampled: # use .iloc
        a = subs.pat_frame_train.iloc[subs.pat_names_train_bins[subs.bin_keys[0]], :]
        b = subs.pat_frame_train.iloc[subs.pat_names_train_bins[subs.bin_keys[-1]], :]
    else: # use .loc
        a = subs.pat_frame_train.loc[subs.pat_names_train_bins[subs.bin_keys[0]], :]
        b = subs.pat_frame_train.loc[subs.pat_names_train_bins[subs.bin_keys[-1]], :]

    t_frame = t_frame_compute(a, b, feat_filter=[])  # ['thickness', 'volume'])
    t_feats = t_frame.columns.tolist()
    t_feats_num = len(t_feats)
    print("computed %d T feats" % t_feats_num)

    # compute f_feats
    f_frame = f_frame_compute(frame=subs.pat_frame_train, y_tgt=subs.pat_frame_train_y,
                              task=subs.tgt_task, feat_filter=[])
    f_feats = f_frame.columns.tolist()
    f_feats_num = len(f_feats)
    print("computed %d F feats" % f_feats_num)
    # compute mi_feats
    mi_frame = mi_frame_compute(frame=subs.pat_frame_train, y_tgt=subs.pat_frame_train_y,
                                task=subs.tgt_task, feat_filter=[])
    mi_feats = mi_frame.columns.tolist()
    mi_feats_num = len(mi_feats)
    print("computed %d MI feats" % mi_feats_num)

    feat_pool_all = t_feats + f_feats + mi_feats
    feat_pool_counts_frame = pd.DataFrame(index=['count'], data=dict(Counter(deepcopy(feat_pool_all))))
    feat_pool_counts_frame.sort_values(by='count', axis=1, ascending=False, inplace=True)

    feat_pool_set = list(set(feat_pool_counts_frame.columns.tolist()))
    print('computed %d pool feats' % len(feat_pool_set))
    all_tgt_results[tgt_name].update({
                                        't_frame': t_frame.transpose(),
                                        'f_frame': f_frame.transpose(),
                                        'mi_frame': mi_frame.transpose(),
                                        'feat_count_frame': feat_pool_counts_frame.transpose()
                                     })
    print('%s : feat_pool computation took %.2f' % (tgt_name, time.time() - zeit))

    # choose training loss and evaluation metric
    xgb_scoring = None
    if subs.tgt_task is 'clf':
        if subs.num_bins is 2:
            lsvm_params = {}
            lsvm_scoring = clf_scorer.balanced_accuracy
            xgb_params = {'objective': 'binary:logistic', 'eval_metric':'auc'}
        elif subs.num_bins > 2:
            lsvm_params = {}
            lsvm_scoring = clf_scorer.balanced_accuracy
            xgb_params = {'objective': 'multi:softmax', 'num_class': subs.num_bins, 'eval_metric': 'mlogloss'}
    elif subs.tgt_task is 'reg':
        lsvm_params = {}
        lsvm_scoring = reg_scorer.neg_mean_square_error
        xgb_params = {}

    # linear non-linear loop
    iteration = {gbl.linear_: {'params': lsvm_params, 'scoring': lsvm_scoring}}#,
                 #gbl.non_linear: {'params': xgb_params, 'scoring': xgb_scoring}
                 #}

    for est_type, value in iteration.items():
        # ml feature selection computation
        zeit = time.time()
        print('%s/%s : starting feat sel RFECV computation' % (tgt_name, est_type))
        feat_sels_rfecv = []
        n_min_feat_rfecv = 10
        # n_max_feat_rfecv = 25
        # potential grid point rfecv loop
        feat_sels_rfecv.append(rfe_cv(est_type=est_type, task=subs.tgt_task, feat_pool=feat_pool_set,
                                      X=subs.pat_frame_train_norm, y=subs.pat_frame_train_y,
                                      cv_folds=subs.cv_folds, n_min_feat=n_min_feat_rfecv,
                                      n_max_feat=None, params=value['params'], scoring=value['scoring']))

        print('%s/%s : feat_sel RFECV computation took %.2f' % (tgt_name, est_type, time.time() - zeit))

        feat_sels = feat_sels_rfecv # potential freq_item_set mining function, include support?
        # naming convention/concept as feat_selections until put into data structure as feature sets
        # (along with hoexter, boedhoe)
        for fsel in feat_sels:
            all_tgt_results[tgt_name][est_type]['feat_set_' + str(feat_sets_count)] = FeatSetResults(fsel)
            feat_sets_count += 1
        all_tgt_results[tgt_name][est_type]['boedhoe'] = FeatSetResults(gbl.boedhoe_feats_Desikan)
        all_tgt_results[tgt_name][est_type]['hoexter'] = FeatSetResults(gbl.hoexter_feats_Desikan)

        # train predict loop for each feat set
        for fset, fresults in all_tgt_results[tgt_name][est_type].items():
            zeit = time.time()
            print('%s/%s/%s : beginning training' % (tgt_name, est_type, fset))
            feat_train = fresults.data['feat_set_list'] + gbl.clin_demog_feats

            est5, val_scores = train(est_type=est_type, task=subs.tgt_task, params=value['params'],
                                     X=subs.pat_frame_train_norm.loc[:, feat_train],
                                     y=subs.pat_frame_train_y,
                                     cv_folds=subs.cv_folds,
                                     scoring=value['scoring']
                                     )
            print('%s/%s/%s : beginning prediction' % (tgt_name, est_type, fset))
            pred_frames, pred_scores, perm_imps = pred(est_type=est_type, task=subs.tgt_task, est5=est5,
                                                       X=subs.pat_frame_test_norm.loc[:, feat_train],
                                                       y=subs.pat_frame_test_y.iloc[:, 0],
                                                       scoring=value['scoring'])

            all_tgt_results[tgt_name][est_type][fset].data.update({
                                                                         'pred_frames': pred_frames,
                                                                         'pred_scores': pred_scores,
                                                                         'conf_interval': conf_interval(pred_scores),
                                                                         'feat_imp_frames': perm_imps,
                                                                         'est5': est5,
                                                                         'train_scores': val_scores
                                                                          })
            all_tgt_results[tgt_name][est_type][fset].sort_prune_pred(thresh=0.5)

            #if no predictions above 0.5 score remove feat set
            # if not all_tgt_results[tgt_name][est_type][fset].data['pred_scores']:
            #     del all_tgt_results[tgt_name][est_type][fset]
            print('%s/%s/%s: training and prediction computation took %.2f' % (tgt_name, est_type, fset,
                                                                               time.time() - zeit))

            if all_tgt_results[tgt_name][est_type][fset].data['pred_scores']:
                lrn_feat_sets[est_type].append(all_tgt_results[tgt_name][est_type][fset].data['feat_set_list'])

            # end train predict loop for each feat set

        # end linear non-linear loop

    # end tgt loop

# feature set mining, frequent item set mining
dataset = lrn_feat_sets[gbl.linear_] + lrn_feat_sets[gbl.non_linear_]
freq_item_sets_frame = freq_item_sets(dataset)
freq_item_sets_list = freq_item_sets_frame.loc[:, 'itemsets'].apply(lambda x: list(x)).tolist()

# collate permutation importance rankings
#feat_all = list(set([item for item in fis for fis in freq_item_set_list]))

#for f in feat_all: for tgt in targets: for
#
#

#
#
#
# # SAVE RESULTS
# print('SAVING RESULTS')
# # str(t_s) + \
# exp_description = '**balRandTest'+str(t_s)+'_RegTrainRest_ClfTrain' + over_samp_names[o_s] + '_' + norm + '_' \
#                   + reg_scorers_names[r_sc] + '_' + clf_scorers_names[c_sc] + '_' + \
#                   'cvFolds' + str(cv_folds) + \
#                   '**t_allRegTrain_DesikanThickVolFeats_TorP'
#
# try:
#     os.mkdir(tgt_name)
# except FileExistsError:
#     pass
#
# bmr = open(tgt_name + '/' + tgt_name + exp_description + '**bmr.pkl', 'wb')
# pickle.dump(gbl.best_models_results, bmr, -1)
# bmr.close()
# try:
#     t_reg_best_score = format(round(gbl.best_models_results[gbl.t_c]['pred_results'].iloc[-1, 0], 2))
# except:
#     t_reg_best_score = -1
# try:
#     t_clf_best_score = format(round(gbl.best_models_results[gbl.t_r]['pred_results'].iloc[-2, 0], 2))
# except:
#     t_clf_best_score = -1
# # write prediction results to excel
# xlsx_name = tgt_name + '/' + tgt_name + exp_description + '**results**' + \
#             'tclf:' + str(t_clf_best_score) +'_' +\
#             'treg:' + str(t_reg_best_score) +'.xlsx'
#
# writer = pd.ExcelWriter(xlsx_name)
# write_report(writer, subs.pat_frame_test_y_clf, subs.pat_frame_test_y_reg)
# frame_name_suffix = '_non-resampled' # SMOTE, ROS, ADASYN
# gbl.t_frame_global.to_excel(writer, 't_frame' + frame_name_suffix)
# gbl.f_frame_global.to_excel(writer, 'f_frame' + frame_name_suffix)
# gbl.mi_frame_global.to_excel(writer, 'mi_frame' + frame_name_suffix)
# feat_pool_counts_frame.to_excel(writer, 'feat_pool_counts')
# writer.save()
# print('SAVED %s' % xlsx_name)
#
#
# t_feats_pats_cons_all = t_frame_compute(subs.pat_frame_train, subs.con_frame, []) # ['thickness', 'volume'])
# writer = pd.ExcelWriter('t_frame_pats_v_cons' + frame_name_suffix)

# print("TOTAL TIME %.2f" % (time.time()-start_time))
