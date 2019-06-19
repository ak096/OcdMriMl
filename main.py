import time
import warnings
import sys
import pickle
import datetime
import os

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, mean_absolute_error
from scipy.stats import sem, t
from scipy import mean

#from pickling import *
from feat_pool_univar import feat_pool_compute
from feat_selection_ml import grid_rfe_cv, freq_item_sets
from dataset import Subs
from results import FeatSetResults
from train_predict import train, pred
from scorers_ import RegScorer, ClfScorer
import gbl

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

feat_sets_results_list = ['freq', 'pred_best', 'tgt_best', 'est_type_best', 'pred_avg', 'pred_ci']

all_fsets_results_clf_frame = pd.DataFrame(index=feat_sets_results_list)
all_fsets_results_clf_dict = {}

all_fsets_results_reg_frame = pd.DataFrame(index=feat_sets_results_list)
all_fsets_results_reg_dict = {}


for idx, tgt_name in enumerate(targets):
    print('init dataset subs')
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

    # univariate feature pool computation:
    print('%s : starting feat_pool computation'% (tgt_name))
    zeit = time.time()
    t_frame, f_frame, mi_frame, feat_pool_counts_frame, feat_pool_set = feat_pool_compute(subs, feat_filter=[])
    print('computed %d pool feats' % len(feat_pool_set))

    all_tgt_results[tgt_name] = {
                                't_frame': t_frame.transpose(),
                                'f_frame': f_frame.transpose(),
                                'mi_frame': mi_frame.transpose(),
                                'feat_count_frame': feat_pool_counts_frame.transpose(),
                                gbl.linear_: {},
                                gbl.non_linear_: {}
                                }

    print('%s : feat_pool computation took %.2f' % (tgt_name, time.time() - zeit))

    # choose training loss and evaluation metric
    lsvm_params = {}
    xgb_params = {}
    scoring = None

    if subs.tgt_task is 'clf':
        scoring = clf_scorer.f1_weighted
        if subs.resampled:
            lsvm_params.update({'class_weight': None})
            #scoring = clf_scorer.accuracy
    if subs.tgt_task is 'reg':
        scoring = reg_scorer.neg_mean_absolute_error

    # linear non-linear loop
    iteration = {gbl.linear_: {'params': lsvm_params},
                 gbl.non_linear_: {'params': xgb_params}
                }

    for est_type, value in iteration.items():
        # ml feature selection computation
        zeit = time.time()
        print('%s/%s : starting feat sel RFECV computation' % (tgt_name, est_type))

        n_min_feat_rfecv = 10
        # n_max_feat_rfecv = 25
        # grid point rfecv loop
        feat_sels_rfecv = grid_rfe_cv(est_type=est_type, task=subs.tgt_task, feat_pool=feat_pool_set,
                                      X=subs.pat_frame_train_norm, y=subs.pat_frame_train_y,
                                      cv_folds=subs.cv_folds, n_min_feat=n_min_feat_rfecv,
                                      n_max_feat=None, params=value['params'], scoring=scoring)

        print('%s/%s : feat_sel RFECV computation took %.2f' % (tgt_name, est_type, time.time() - zeit))

        # feature set mining, frequent item set mining
        dataset = feat_sels_rfecv
        freq_item_sets_frame = freq_item_sets(dataset, min_support=0.75)
        freq_item_sets_list = freq_item_sets_frame.loc[:, 'itemsets'].apply(lambda x: list(x)).tolist()
        feat_sels = freq_item_sets_list # potential freq_item_set mining function, include support?
        # naming convention/concept : feat_selections until put into data structure as feature sets
        # (along with hoexter, boedhoe)
        for fsel in feat_sels:
            all_tgt_results[tgt_name][est_type]['featset_' + str(feat_sets_count)] = FeatSetResults(fsel)
            feat_sets_count += 1
        all_tgt_results[tgt_name][est_type]['boedhoe'] = FeatSetResults(gbl.boedhoe_feats_Desikan)
        all_tgt_results[tgt_name][est_type]['hoexter'] = FeatSetResults(gbl.hoexter_feats_Desikan)


        # train predict loop for each feat set
        for fset, fresults in all_tgt_results[tgt_name][est_type].items():
            zeit = time.time()
            print('%s/%s/%s : training' % (tgt_name, est_type, fset))
            feat_train = fresults.data['feat_set_list'] + gbl.clin_demog_feats

            est5, val_scores = train(est_type=est_type, task=subs.tgt_task,
                                     X=subs.pat_frame_train_norm.loc[:, feat_train],
                                     y=subs.pat_frame_train_y,
                                     cv_folds=subs.cv_folds,
                                     params=value['params'],
                                     scoring=scoring
                                     )
            print('%s/%s/%s : predicting' % (tgt_name, est_type, fset))
            pred_frames, pred_scores, perm_imps = pred(est_type=est_type, task=subs.tgt_task, est5=est5,
                                                       X=subs.pat_frame_test_norm.loc[:, feat_train],
                                                       y=subs.pat_frame_test_y.iloc[:, 0],
                                                       scoring=scoring)

            fresults.data.update({
                                 'pred_frames': pred_frames,
                                 'pred_scores': pred_scores,
                                 'scoring': scoring,
                                 'conf_interval': conf_interval(pred_scores),
                                 'feat_imp_frames': perm_imps,
                                 'est5': est5,
                                 'train_scores': val_scores
                                  })

            if subs.tgt_task is gbl.clf:
                fresults.sort_prune_pred(pred_score_thresh=0.5)
                if fresults.data['pred_scores']:
                    exists = False
                    for k, v in all_fsets_results_clf_dict.items():
                        if fresults.data['feat_set_list'] == v['feat_set_list']:
                            all_fsets_results_clf_frame.loc['freq', k] += 1
                            all_fsets_results_clf_dict[k]['preds'].append(fresults.data['pred_scores'][0])
                            if fresults.data['pred_scores'][0] > all_fsets_results_clf_frame.loc['pred_best', k]:
                                all_fsets_results_clf_frame.loc['pred_best', k] = fresults.data['pred_scores'][0]
                                all_fsets_results_clf_frame.loc['tgt_best', k] = tgt_name
                                all_fsets_results_clf_frame.loc['est_type_best', k] = est_type
                            exists = True
                            break
                    if not exists: # add fset into results
                        all_fsets_results_clf_frame[fset] = [1, fresults.data['pred_scores'][0], tgt_name, est_type,
                                                             -1, -1]
                        all_fsets_results_clf_dict[fset] = {'feat_set_list': fresults.data['feat_set_list'],
                                                            'est_best': fresults.data['est5'][0],
                                                            'pred_frame_best': fresults.data['pred_frames'][0],
                                                            'preds': [fresults.data['pred_scores'][0]]}
            elif subs.tgt_task is gbl.reg:
                fresults.sort_prune_pred(pred_score_thresh=-7.0)
                if fresults.data['pred_scores']:
                    exists = False
                    for k, v in all_fsets_results_reg_dict.items():
                        if fresults.data['feat_set_list'] == v['feat_set_list']:
                            all_fsets_results_reg_frame.loc['freq', k] += 1
                            all_fsets_results_reg_dict[k]['preds'].append(fresults.data['pred_scores'][0])
                            if fresults.data['pred_scores'][0] > all_fsets_results_reg_frame.loc['pred_best', k]:
                                all_fsets_results_reg_frame.loc['pred_best', k] = fresults.data['pred_scores'][0]
                                all_fsets_results_reg_frame.loc['tgt_best', k] = tgt_name
                                all_fsets_results_reg_frame.loc['est_type_best', k] = est_type
                            exists = True
                            break
                    if not exists:  # add fset into results
                        all_fsets_results_reg_frame[fset] = [1, fresults.data['pred_scores'][0], tgt_name, est_type,
                                                             -1, -1]
                        all_fsets_results_reg_dict[fset] = {'feat_set_list': fresults.data['feat_set_list'],
                                                            'est_best': fresults.data['est5'][0],
                                                            'pred_frame_best': fresults.data['pred_frames'][0],
                                                            'preds': [fresults.data['pred_scores'][0]]}

            print('%s/%s/%s: train and predict took %.2f' % (tgt_name, est_type, fset,
                                                                               time.time() - zeit))

            # end train predict loop for each feat set

        # end linear non-linear loop

    # end tgt loop


# collate permutation importance rankings
# purge all
#feat_all = list(set([item for item in fis for fis in freq_item_set_list]))

#for f in feat_all: for tgt in targets: for

#
#
#
# # SAVE RESULTS
# print('SAVING RESULTS')

# exp_description = '**balRandTest'+str(t_s)+'_RegTrainRest_ClfTrain' + over_samp_names[o_s] + '_' + norm + '_' \
#                   + reg_scorers_names[r_sc] + '_' + clf_scorers_names[c_sc] + '_' + \
#                   'cvFolds' + str(cv_folds) + \
#                   '**t_allRegTrain_DesikanThickVolFeats_TorP'
#
#
# # write prediction results to excel
# xlsx_name =
#
# writer = pd.ExcelWriter(xlsx_name)
# feat_pool_counts_frame.to_excel(writer, 'feat_pool_counts')
# writer.save()
# print('SAVED %s' % xlsx_name)

print("TOTAL TIME %.2f" % (time.time()-start_time))
