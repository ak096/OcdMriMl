import time
import warnings
import sys
import pickle
import datetime
import os
import resource

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

#from pickling import *
from feat_pool_univar import feat_pool_compute
from feat_selection_ml import grid_rfe_cv, freq_item_sets_compute, feat_perm_imp_compute
from dataset import Subs
from results import update_results #, FeatSetResults
from train_predict import train, pred
from scorers_ import RegScorer, ClfScorer
import gbl

start_time = time.time()
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

sys.setrecursionlimit(1500)
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

# seed = 7
# np.random.seed(seed)

targets = [
           #'YBOCS_reg',
           #'YBOCS_class2',
            #'obs_class3',
            #'com_class3',
           'YBOCS_class3',
           'YBOCS_class4'
          ]

#all_tgt_results = {}

clf_scorer = ClfScorer()
reg_scorer = RegScorer()

fsets_count = 1

fsets_results_frame_idx = ['freq', 'pred_best', 'tgt_best', 'est_type_best', 'pred_avg', 'pred_ci']

all_fsets_results_clf_frame = pd.DataFrame(index=fsets_results_frame_idx)
all_fsets_results_clf_dict = {}

all_fsets_results_reg_frame = pd.DataFrame(index=fsets_results_frame_idx)
all_fsets_results_reg_dict = {}


for idx, tgt_name in enumerate(targets):

    if '_ROS' in tgt_name:
        subs.resample(over_sampler='ROS')
    elif '_SVMSMOTE' in tgt_name:
        subs.resample(over_sampler='SVMSMOTE')
    else:
        print('%s: initialize dataset (subs): ' % (tgt_name))
        subs = Subs(tgt_name=tgt_name, test_size=0.15)
        if subs.imbalanced_classes:
            targets.insert(idx+1, tgt_name + '_ROS')
            targets.insert(idx+2, tgt_name + '_SVMSMOTE')

    n = 0
    norm = gbl.normType_list[n]

    # univariate feature pool computation:
    zeit = time.time()
    t_frame, f_frame, mi_frame, feat_pool_counts_frame, feat_pool_set = feat_pool_compute(tgt_name, subs,
                                                                                          feat_filter=[])
    feat_pool_set_num = len(feat_pool_set)
    # all_tgt_results[tgt_name] = {
    #                             #'t_frame': t_frame.transpose(),
    #                             #'f_frame': f_frame.transpose(),
    #                             #'mi_frame': mi_frame.transpose(),
    #                             #'feat_count_frame': feat_pool_counts_frame.transpose(),
    #                             gbl.linear_: {},
    #                             gbl.non_linear_: {}
    #                             }

    print('%s: computed feat_pool: %d' % (tgt_name, feat_pool_set_num))

    # choose training loss and evaluation metric
    lsvm_params = {}
    xgb_params = {}
    scoring = None
    thresh = None
    if subs.tgt_task is 'clf':
        scoring = clf_scorer.recall_weighted
        thresh = 0.5
        if subs.resampled:
            lsvm_params.update({'class_weight': None})
            #scoring = clf_scorer.accuracy
    if subs.tgt_task is 'reg':
        scoring = reg_scorer.neg_mean_absolute_error
        thres = gbl.YBOCS_std

    # linear non-linear loop
    iteration = {gbl.linear_: {'params': lsvm_params},
                 gbl.non_linear_: {'params': xgb_params}
                }

    for est_type, params in iteration.items():
        # ml feature selection computation
        zeit = time.time()
        print('%s/%s: RFECV starting with feat pool:' % (tgt_name, est_type), feat_pool_set_num)

        n_min_feat_rfecv = 5
        n_max_feat_rfecv = None
        # grid point rfecv loop
        feat_sels_rfecv = grid_rfe_cv(tgt_name=tgt_name, est_type=est_type, task=subs.tgt_task, feat_pool=feat_pool_set,
                                      X=subs.pat_frame_train_norm, y=subs.pat_frame_train_y,
                                      cv_folds=subs.cv_folds, n_min_feat=n_min_feat_rfecv,
                                      n_max_feat=n_max_feat_rfecv, params=params['params'], scoring=scoring)

        print('%s/%s: RFECV took %.2f sec' % (tgt_name, est_type, time.time() - zeit))
        # frequent item set mining
        print('%s/%s: FIS starting' % (tgt_name, est_type))
        freq_item_sets_list = freq_item_sets_compute(feat_sels_rfecv, min_sup=0.90)
        #freq_item_sets_list = [fis for fis in freq_item_sets_list if len(fis) >= 2]
        print('%s/%s: FIS resulted in %d sets' % (tgt_name, est_type, len(freq_item_sets_list)))

        feat_sels = freq_item_sets_list

        # naming convention/concept : feat_selections until put into data structure as feature sets
        # (along with hoexter, boedhoe)
        # for fsel in feat_sels:
        #     all_tgt_results[tgt_name][est_type]['fset_' + str(fsets_count)] = FeatSetResults(fsel)
        #     fsets_count += 1
        # all_tgt_results[tgt_name][est_type]['boedhoe_' + str(fsets_count)] = FeatSetResults(gbl.boedhoe_feats_Desikan)
        # fsets_count += 1
        # all_tgt_results[tgt_name][est_type]['hoexter_' + str(fsets_count)] = FeatSetResults(gbl.hoexter_feats_Desikan)
        fset_dict = {}
        for fsel in feat_sels:
            fset_dict['fset_' + str(fsets_count)] = fsel
            fsets_count += 1
        fset_dict['boedhoe_' + str(fsets_count)] = gbl.boedhoe_feats_Desikan
        fsets_count += 1
        fset_dict['hoexter_' + str(fsets_count)] = gbl.hoexter_feats_Desikan
        # train predict loop for each feat set
        #for fset, fset_results in all_tgt_results[tgt_name][est_type].items():
        for fset, fset_list in fset_dict.items():
            zeit = time.time()
            print('%s/%s/%s/%d: scoring used:' % (tgt_name, est_type, fset, fsets_count), scoring)

            #feat_train = fset_results.data['fset_list'] + gbl.clin_demog_feats
            feat_train = fset_list + gbl.clin_demog_feats

            ests, train_scores = train(est_type=est_type, task=subs.tgt_task,
                                       X=subs.pat_frame_train_norm.loc[:, feat_train],
                                       y=subs.pat_frame_train_y,
                                       cv_folds=subs.cv_folds,
                                       params=params['params'],
                                       scoring=scoring,
                                       thresh=thresh
                                       )
            print('%s/%s/%s/%d: trained cv over grid:' % (tgt_name, est_type, fset, fsets_count), train_scores)

            pred_scores, pred_frames = pred(est_type=est_type, task=subs.tgt_task, ests=ests,
                                            X=subs.pat_frame_test_norm.loc[:, feat_train],
                                            y=subs.pat_frame_test_y.iloc[:, 0],
                                            scoring=scoring,
                                            thresh=thresh
                                           )
            print('%s/%s/%s/%d: predicted:' % (tgt_name, est_type, fset, fsets_count), pred_scores)

            # fset_results.data.update({
            #                      #'pred_frames': pred_frames,
            #                      'pred_scores': pred_scores,
            #                      'scoring': scoring,
            #                      'conf_interval': conf_interval(pred_scores)[1],
            #                      'feat_imp_frames': perm_imps,
            #                      'ests': ests,
            #                      'train_scores': train_scores
            #                       })
            print('%s/%s/%s/%d: train and predict took %.2f' % (tgt_name, est_type, fset, fsets_count,
                                                                time.time() - zeit))
            tp = sorted(list(zip(pred_scores, pred_frames, ests)), key=lambda x: x[0], reverse=True)[0]
            psb, pfb, eb = tp
            print('%s/%s/%s/%d: found best pred score:' % (tgt_name, est_type, fset, fsets_count), psb)
            if psb <= thresh:
                print('%s/%s/%s/%d: skipping:' % (tgt_name, est_type, fset, fsets_count))
            else:

                fset_results = {'pred_score_best': psb, 'pred_frame_best': pfb, 'fset_list': fset_list, 'est_best': eb}
                if subs.tgt_task is gbl.clf:
                    #fset_results.sort_prune_pred(pred_score_thresh=0.5)
                    # print('%s/%s/%s/%d: sorted and pruned to:' % (tgt_name, est_type, fset, fsets_count),
                    #       fset_results.data['pred_scores'])
                    all_fsets_results_clf_frame, all_fsets_results_clf_dict = update_results(tgt_name, est_type, fsets_count,
                                                                                             fset, fset_results,
                                                                                             all_fsets_results_clf_frame,
                                                                                             all_fsets_results_clf_dict)
                elif subs.tgt_task is gbl.reg:
                    #fset_results.sort_prune_pred(pred_score_thresh=-gbl.YBOCS_std)
                    #print('%s/%s/%s/%d: sorted and pruned to:' % (tgt_name, est_type, fset, fsets_count),
                    #      fset_results.data['pred_scores'])
                    all_fsets_results_reg_frame, all_fsets_results_reg_dict = update_results(tgt_name, est_type, fsets_count,
                                                                                             fset, fset_results,
                                                                                             all_fsets_results_reg_frame,
                                                                                             all_fsets_results_reg_dict)

            print()

            # end train predict loop for each feat set

        # end linear non-linear loop

    # end tgt loop
fpi_results_clf_dict = feat_perm_imp_compute(gbl.fpis_clf)
fpi_results_reg_dict = feat_perm_imp_compute(gbl.fpis_reg)

# collate permutation importance rankings
feat_perm_imp_results_clf_frame = pd.DataFrame().from_dict(fpi_results_clf_dict)
feat_perm_imp_results_reg_frame = pd.DataFrame().from_dict(fpi_results_reg_dict)


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
