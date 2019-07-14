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
from feat_selection_ml import grid_rfe_cv, compute_fqis_pyfpgrowth_dict,  largest_common_subsets
from dataset import Subs
from results import update_fset_results, compute_fset_results_frame, compute_fpi_results_dict #, FeatSetResults
from train_predict import train, pred
from scorers_ import RegScorer, ClfScorer
import gbl

start_time = time.time()
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

sys.setrecursionlimit(10**8)
resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

# seed = 7
# np.random.seed(seed)

targets = [
'YBOCS_class2',
#'obs_class3',
#'com_class3',
'YBOCS_class3',
'YBOCS_class4',
'YBOCS_reg',
          ]

tgt_dict = {}

clf_scorer = ClfScorer()
reg_scorer = RegScorer()

fsets_count_int = 0

fsets_results_frame_idx = ['freq', 'pred_best', 'tgt_best', 'est_type_best', 'pred_avg', 'pred_ci']

fsets_results_clf_frame = pd.DataFrame(index=fsets_results_frame_idx)
fsets_results_clf_dict = {}
fsets_names_clf_frame = pd.DataFrame()

fsets_results_reg_frame = pd.DataFrame(index=fsets_results_frame_idx)
fsets_results_reg_dict = {}
fsets_names_reg_frame = pd.DataFrame()

# settings for experiment
atlas = 'DesiDest' #'Desikan' or 'Destrieux' or 'DesiDest'
min_support = 0.9

for idx, tgt_name in enumerate(targets):

    if '_ROS' in tgt_name:
        subs.resample(over_sampler='ROS')
    elif '_SMOTE' in tgt_name:
        subs.resample(over_sampler='SMOTE')
    else:
        print('%s: initialize dataset (subs): ' % (tgt_name))
        subs = Subs(tgt_name=tgt_name, test_size=0.15)
        if subs.imbalanced_classes:
            targets.insert(idx+1, tgt_name + '_ROS')
            targets.insert(idx+2, tgt_name + '_SMOTE')

    #n = 0
    #norm = gbl.normType_list[n]

    # univariate feature pool computation:
    zeit = time.time()
    feat_picker = gbl.atlas_dict[atlas]
    # feat_picker used with entire atlas but could be list of any substrings in feats to pick (e.g. '_volume')
    t_frame, f_frame, mi_frame, feat_pool_counts_frame, feat_pool_set = feat_pool_compute(tgt_name=tgt_name, subs=subs,
                                                                                          feat_picker=feat_picker)
    feat_pool_set_num = len(feat_pool_set)
    tgt_dict[tgt_name] = {
                                't_frame': t_frame.transpose(),
                                'f_frame': f_frame.transpose(),
                                'mi_frame': mi_frame.transpose(),
                                'feat_count_frame': feat_pool_counts_frame.transpose(),
                                'subs': subs
                                }

    print('%s: computed feat_pool: %d' % (tgt_name, feat_pool_set_num))

    # choose training loss and evaluation metric
    lsvm_params = {}
    xgb_params = {}
    scoring = None
    thresh = None
    if subs.tgt_task is gbl.clf:
        scoring = clf_scorer.recall_weighted
        thresh = 0.5
        if subs.resampled:
            lsvm_params.update({'class_weight': None})
            #scoring = clf_scorer.accuracy
    if subs.tgt_task is gbl.reg:
        scoring = reg_scorer.neg_mean_absolute_error
        thresh = -gbl.YBOCS_reg_std

    # linear non-linear loop
    iteration = {gbl.linear_: {'params': lsvm_params},
                 gbl.non_linear_: {'params': xgb_params}
                }

    for est_type, params in iteration.items():
        # ml feature selection computation
        zeit = time.time()
        print('%s/%s: RFECV starting with feat pool:' % (tgt_name, est_type), feat_pool_set_num)

        n_min_feat_rfecv = 10
        n_max_feat_rfecv = None
        # grid point rfecv loop
        feat_sels_rfecv = grid_rfe_cv(tgt_name=tgt_name, est_type=est_type, task=subs.tgt_task, feat_pool=feat_pool_set,
                                      X=subs.pat_frame_train_norm, y=subs.pat_frame_train_y,
                                      cv_folds=subs.cv_folds, n_min_feat=n_min_feat_rfecv,
                                      n_max_feat=n_max_feat_rfecv, params=params['params'], scoring=scoring)

        print('%s/%s: RFECV took %.2f sec' % (tgt_name, est_type, time.time() - zeit))
        # # frequent item set mining
        # print('%s/%s: FIS starting' % (tgt_name, est_type))
        # freq_item_sets_list = freq_item_sets_compute(feat_sels_rfecv, min_sup=0.90)
        #
        # print('%s/%s: FIS resulted in %d sets' % (tgt_name, est_type, len(freq_item_sets_list)))
        # largest common subsets
        print('%s/%s: LCS starting' % (tgt_name, est_type))
        lcs_list = largest_common_subsets(super_ilists=feat_sels_rfecv, min_sup=round(min_support * len(feat_sels_rfecv)))
        print('%s/%s: LCS resulted in %d sets' % (tgt_name, est_type, len(lcs_list)))
        feat_sels = lcs_list

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
            fsets_count_int += 1
            fset_dict['fset_' + str(fsets_count_int)] = fsel

        names = ['hoexter_', 'boedhoe_csc_', 'boedhoe_c_', 'boedhoe_sc_']
        for i, exp_fset in enumerate(gbl.h_b_expert_fsets[atlas]):
            fsets_count_int += 1
            fset_dict[names[i] + str(fsets_count_int)] = exp_fset

        # train predict loop for each feat set
        #for fset, fset_results in all_tgt_results[tgt_name][est_type].items():
        for fset_name, fset_list in fset_dict.items():
            zeit = time.time()
            print('%s/%s/%s/%d: scoring used:' % (tgt_name, est_type, fset_name, fsets_count_int), scoring)

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
            print('%s/%s/%s/%d: trained cv over grid:' % (tgt_name, est_type, fset_name, fsets_count_int), train_scores)

            pred_scores, pred_frames = pred(est_type=est_type, task=subs.tgt_task, ests=ests,
                                            X=subs.pat_frame_test_norm.loc[:, feat_train],
                                            y=subs.pat_frame_test_y.iloc[:, 0],
                                            scoring=scoring,
                                            thresh=thresh
                                           )
            print('%s/%s/%s/%d: predicted:' % (tgt_name, est_type, fset_name, fsets_count_int), pred_scores)

            # fset_results.data.update({
            #                      #'pred_frames': pred_frames,
            #                      'pred_scores': pred_scores,
            #                      'scoring': scoring,
            #                      'conf_interval': conf_interval(pred_scores)[1],
            #                      'feat_imp_frames': perm_imps,
            #                      'ests': ests,
            #                      'train_scores': train_scores
            #                       })
            print('%s/%s/%s/%d: train and predict took %.2f' % (tgt_name, est_type, fset_name, fsets_count_int,
                                                                time.time() - zeit))
            tp = sorted(list(zip(pred_scores, pred_frames, ests)), key=lambda x: x[0], reverse=True)[0]
            psb, pfb, eb = tp
            print('%s/%s/%s/%d: found best pred score:' % (tgt_name, est_type, fset_name, fsets_count_int), psb)
            if psb <= thresh:
                print('%s/%s/%s/%d: below thresh, skipping update' % (tgt_name, est_type, fset_name, fsets_count_int))
            else:

                fset_results = {'pred_score_best': psb, 'pred_frame_best': pfb, 'fset_list': fset_list, 'est_best': eb}
                if subs.tgt_task is gbl.clf:
                    #fset_results.sort_prune_pred(pred_score_thresh=0.5)
                    # print('%s/%s/%s/%d: sorted and pruned to:' % (tgt_name, est_type, fset, fsets_count),
                    #       fset_results.data['pred_scores'])
                    fsets_results_clf_frame, \
                    fsets_results_clf_dict, \
                    fsets_names_clf_frame = update_fset_results(tgt_name, est_type, fsets_count_int, fset_name, fset_results,
                                                                fsets_results_clf_frame,
                                                                fsets_results_clf_dict,
                                                                fsets_names_clf_frame)
                elif subs.tgt_task is gbl.reg:
                    #fset_results.sort_prune_pred(pred_score_thresh=-gbl.YBOCS_std)
                    #print('%s/%s/%s/%d: sorted and pruned to:' % (tgt_name, est_type, fset, fsets_count),
                    #      fset_results.data['pred_scores'])
                    fsets_results_reg_frame, \
                    fsets_results_reg_dict, \
                    fsets_names_reg_frame = update_fset_results(tgt_name, est_type, fsets_count_int, fset_name, fset_results,
                                                                fsets_results_reg_frame,
                                                                fsets_results_reg_dict,
                                                                fsets_names_reg_frame)

            print()
            #clear cache due to memory errors
            ests, train_scores, pred_scores, pred_frames = [None, None, None, None]
            # end train predict loop for each feat set
        # clear cache
        feat_sels_rfecv = None
        # end linear non-linear loop
    #clear cache for tgt loop
    #subs = None
    # end tgt loop
#fpi_results_clf_dict = compute_fpi_results_dict(gbl.fpi_clf)
#fpi_results_reg_dict = compute_fpi_results_dict(gbl.fpi_reg)

# collate permutation importance rankings
#fpi_results_clf_frame = pd.DataFrame().from_dict(fpi_results_clf_dict)
#fpi_results_reg_frame = pd.DataFrame().from_dict(fpi_results_reg_dict)

# compute pred_ci, pred_avg and sort
fsets_results_clf_frame = compute_fset_results_frame(fsets_results_clf_frame, fsets_results_clf_dict)
fsets_results_reg_frame = compute_fset_results_frame(fsets_results_reg_frame, fsets_results_reg_dict)

# sort results
fsets_results_clf_frame.sort_values(by='pred_best', axis=1, ascending=False, inplace=True)
fsets_results_reg_frame.sort_values(by='pred_best', axis=1, ascending=False, inplace=True)

fsets_names_clf_frame = fsets_names_clf_frame.reindex(columns=fsets_results_clf_frame.columns.tolist())
fsets_names_reg_frame = fsets_names_reg_frame.reindex(columns=fsets_results_reg_frame.columns.tolist())

#fpi_results_clf_frame.sort_values(by='perm_imp_high', axis=1, ascending=False, inplace=True)
#fpi_results_reg_frame.sort_values(by='perm_imp_high', axis=1, ascending=False, inplace=True)


# SAVE RESULTS
def save_results():
    print('SAVING RESULTS')

    exp_description = 'atlas_{}_maxgridpoints_{}_minsupport_{}_geqbg_fpi.xlsx'.format(atlas, gbl.grid_space_size, min_support)

    # write prediction results to excel
    xlsx_name = exp_description

    writer = pd.ExcelWriter(xlsx_name)
    fsets_results_clf_frame.to_excel(writer, 'fsets_results_clf')
    fsets_names_clf_frame.to_excel(writer, 'fsets_names_clf')
    #fpi_results_clf_frame.to_excel(writer, 'fimps_clf')

    fsets_results_reg_frame.to_excel(writer, 'fsets_results_reg')
    fsets_names_reg_frame.to_excel(writer, 'fsets_names_reg')
    #fpi_results_reg_frame.to_excel(writer, 'fimps_reg')

    writer.save()
    print('SAVED %s' % xlsx_name)

    # save_list = [fsets_results_clf_dict, fsets_results_reg_dict, fsets_count_int, tgt_univar_results_dict]
    # for l in save_list:
    #     my_var_name = [k for k, v in locals().items() if v == l and k is not 'l'][0]
    #     with open(atlas+'_'+my_var_name+'.pickle', 'wb') as handle:
    #         pickle.dump(l, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('{}_{}.pickle'.format(atlas, 'fsets_results_clf_dict'), 'wb') as handle:
        pickle.dump(fsets_results_clf_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('{}_{}.pickle'.format(atlas, 'fsets_results_reg_dict'), 'wb') as handle:
        pickle.dump(fsets_results_reg_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('{}_{}.pickle'.format(atlas, 'fsets_count_int'), 'wb') as handle:
        pickle.dump(fsets_count_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('{}_{}.pickle'.format(atlas, 'tgt_dict'), 'wb') as handle:
        pickle.dump(tgt_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('{}_{}.pickle'.format(atlas, 'fpi_clf_dict'), 'wb') as handle:
    #     pickle.dump(gbl.fpi_clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open('{}_{}.pickle'.format(atlas, 'fpi_reg_dict'), 'wb') as handle:
    #     pickle.dump(gbl.fpi_reg, handle, protocol=pickle.HIGHEST_PROTOCOL)


save_results()

print("TOTAL TIME %.2f" % (time.time()-start_time))
