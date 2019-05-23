import numpy as np
import pandas as pd
import gbl
from fs_read import fs_data_collect
from estimators import regress, classify
from scale import scale, testSet_scale
from prediction_reporting import predict_report, write_report
from pat_sets import *
from sklearn.model_selection import train_test_split
import time
import pickle
from pickling import *
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
import copy
import sys
import random
from sklearn.feature_selection import RFECV
from feat_select import t_frame_compute
import xgboost
from sklearn.svm import SVC
from numpy.random import randint
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from trained_models_analysis import models_to_results
from select_pat_names_test_clf import select_pat_names_test_clf
from set_ops import powerset
from RFECV_best_feats import rfecv
from dataset import Subs
start_time = time.time()


warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

# non-ocd (0-9) mild (10-20), moderate (21-30), severe (31-40) Okasha et. al. (2000)

gbl.init_globals()

cv_folds = 7

# seed = 7
# np.random.seed(seed)


reg_scorers = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_log_error', None]
reg_scorers_names = ['ev', 'nmae', 'nmsle', 'None']
r_sc = 1
reg_scoring = reg_scorers[r_sc]

clf_scorers = ['balanced_accuracy', 'accuracy']
clf_scorers_names = ['bac', 'ac']
c_sc = 0
clf_scoring = clf_scorers[c_sc]

iteration = {'n': [0],
             'clf_targets': ['YBOCS_class2_scorerange',
                             #'obs_class3_scorerange',
                             #'com_class3_scorerange',
                             'YBOCS_class3_scorerange',
                             #'YBOCS_class4_scorerange'
                            ]
             }

gbl.feat_sets_best_train[gbl.h_r] = gbl.hoexter_feats_FS + gbl.demo_clin_feats
gbl.feat_sets_best_train[gbl.h_c] = gbl.hoexter_feats_FS + gbl.demo_clin_feats
gbl.feat_sets_best_train[gbl.b_r] = gbl.boedhoe_feats_FS + gbl.demo_clin_feats
gbl.feat_sets_best_train[gbl.b_c] = gbl.boedhoe_feats_FS + gbl.demo_clin_feats

for clf_tgt in iteration['clf_targets']:
    subs = Subs(clf_tgt=clf_tgt, test_size=0.18, over_sampler='SMOTE')

    t_reg_models_all = []
    t_clf_models_all = []

    # expert-picked-feature-based models for regression and classification
    hoexter_reg_models_all = []
    hoexter_clf_models_all = []
    boedhoe_reg_models_all = []
    boedhoe_clf_models_all = []

    brfc_models_all = []

    for n in iteration['n']: # standard, minmax, (robust-quantile-based) normalizations of input data
        t0 = time.time()
        norm = gbl.normType_list[n]

        # compute t_feats
        t_frame = t_frame_compute(subs.pat_frame_train_reg, subs.con_frame, ['thickness', 'volume'])
        gbl.t_frame_global = t_frame
        t_feats_all = t_frame.columns.tolist()
        t_feats_all_num = t_frame.shape[1]
        t_feats_train_all_num = t_feats_all_num + len(gbl.demo_clin_feats)
        print("FINISHED COMPUTING %d T VALUES" % t_feats_all_num)

        print(pd.DataFrame(data=t_feats_all))
        t_feats_idx_powerset = powerset(range(t_feats_all_num))  # powerset of indices without empty set
        print('COMPUTING %d SUBSETS of T_FEATS' % len(t_feats_idx_powerset))
        for t_feats_idx_subset in t_feats_idx_powerset:
            t0 = time.time()
            t_feats_subset = [t_feats_all[i] for i in t_feats_idx_subset]
            t_feats_train = t_feats_subset + gbl.demo_clin_feats
            t_feats_train_num = len(t_feats_train)

            print("COMPUTING %d / %d FEATS W/ NORM %s" % (t_feats_train_num,
                                                          t_feats_train_all_num,
                                                          norm))
            # t_reg_models_all += regress(subs.pat_frame_train_reg_norms[n][t_feats_train],
            #                             subs.pat_frame_train_y_reg, cv_folds,
            #                             reg_scoring, n, glob.t_r, t_feats_train_all_num, t_feats_idx_subset)

            t_clf_models_all += classify(subs.pat_frame_train_clf_norms[n][t_feats_train],
                                         subs.pat_frame_train_y_clf, cv_folds,
                                         clf_scoring, n, gbl.t_c, t_feats_train_all_num, t_feats_idx_subset)

            print("%s: Running brfc: %d OF %d FEATS" % (gbl.t_c, t_feats_train_num, t_feats_train_all_num))
            brfc_models_all.append([BalancedRandomForestClassifier(n_estimators=500,
                                                                   random_state=np.random.RandomState(),
                                                                   n_jobs=-1,
                                                                   class_weight='balanced').fit(subs.pat_frame_train_reg_norms[n][t_feats_train],
                                                                                                subs.pat_frame_train_y_clf_list[0]),
                                    t_feats_train
                                    ])
            print("FINISHED %d / %d FEATS W/ NORM %s TOOK %.2f SEC" % (t_feats_train_num,
                                                                       t_feats_train_all_num,
                                                                       norm,
                                                                       time.time() - t0))

        # end for t_feats


        # RFECV
        #########rfecv(pat_frame_...)

        t1 = time.time()
        print("HOEXTER Regression with norm " + norm)

        # hoexter_reg_models_all += regress(pat_frame_train_reg_norms[n][glob.feat_sets_best_train[glob.h_r]],
        #                                   pat_frame_train_y_reg, cv_folds, reg_scoring, n, glob.h_r,
        #                                   len(glob.feat_sets_best_train[glob.h_r]))

        hoexter_clf_models_all += classify(subs.pat_frame_train_clf_norms[n][gbl.feat_sets_best_train[gbl.h_c]],
                                           subs.pat_frame_train_y_clf, cv_folds, clf_scoring, n, gbl.h_c,
                                           len(gbl.feat_sets_best_train[gbl.h_c]))
        # print("BOEDHOE Regression with norm " + norm)

        # boedhoe_reg_models_all += regress(pat_frame_train_reg_norms[n][glob.feat_sets_best_train[glob.b_r]],
        #                                   pat_frame_train_y_reg, cv_folds, reg_scoring, n, glob.b_r,
        #                                   len(glob.feat_sets_best_train[glob.b_r]))

        boedhoe_clf_models_all += classify(subs.pat_frame_train_clf_norms[n][gbl.feat_sets_best_train[gbl.b_c]],
                                           subs.pat_frame_train_y_clf, cv_folds, clf_scoring, n, gbl.b_c,
                                           len(gbl.feat_sets_best_train[gbl.b_c]))

        print("HOEXTER and BOEDHOE EST W/ NORM %s TOOK %.2f SEC" % (norm, time.time()-t1))

    # end for n norm

    # find best trained models and prediction results
    models_all = {gbl.h_r: hoexter_reg_models_all, gbl.h_c: hoexter_clf_models_all,
                  gbl.b_r: boedhoe_reg_models_all, gbl.b_c: boedhoe_clf_models_all,
                  gbl.t_r: t_reg_models_all, gbl.t_c: t_clf_models_all}

    models_to_results(models_all, subs.pat_frame_test_reg_norms, subs.pat_frame_test_clf_norms,
                      subs.pat_frame_test_y_reg, subs.pat_frame_test_y_clf, reg_scoring)

    # combine best t feats with boedhoe and hoexter
    gbl.feat_sets_best_train[gbl.h_t_r] = gbl.feat_sets_best_train[gbl.t_r] + gbl.hoexter_feats_FS
    gbl.feat_sets_best_train[gbl.h_t_c] = gbl.feat_sets_best_train[gbl.t_c] + gbl.hoexter_feats_FS
    gbl.feat_sets_best_train[gbl.b_t_r] = gbl.feat_sets_best_train[gbl.t_r] + gbl.boedhoe_feats_FS
    gbl.feat_sets_best_train[gbl.b_t_c] = gbl.feat_sets_best_train[gbl.t_c] + gbl.boedhoe_feats_FS

    hoexter_t_reg_models_all = []
    hoexter_t_clf_models_all = []
    boedhoe_t_reg_models_all = []
    boedhoe_t_clf_models_all = []

    # hoexter_t_reg_models_all = regress(pat_frame_train_reg_norms[n][glob.feat_sets_best_train[glob.h_t_r]],
    #                                    pat_frame_train_y_reg, cv_folds, reg_scoring, n, glob.h_t_r,
    #                                    len(glob.feat_sets_best_train[glob.h_t_r]))

    hoexter_t_clf_models_all = classify(subs.pat_frame_train_clf_norms[n][gbl.feat_sets_best_train[gbl.h_t_c]],
                                        subs.pat_frame_train_y_clf, cv_folds, clf_scoring, n, gbl.h_t_c,
                                        len(gbl.feat_sets_best_train[gbl.h_t_c]))
    # print("BOEDHOE Regression with norm " + norm)

    # boedhoe_t_reg_models_all = regress(pat_frame_train_reg_norms[n][glob.feat_sets_best_train[glob.b_t_r]],
    #                                    pat_frame_train_y_reg, cv_folds, reg_scoring, n, glob.b_t_r,
    #                                    len(glob.feat_sets_best_train[glob.b_t_r]))

    boedhoe_t_clf_models_all = classify(subs.pat_frame_train_clf_norms[n][gbl.feat_sets_best_train[gbl.b_t_c]],
                                        subs.pat_frame_train_y_clf, cv_folds, clf_scoring, n, gbl.b_t_c,
                                        len(gbl.feat_sets_best_train[gbl.b_t_c]))

    models2_all = {gbl.h_t_r: hoexter_t_reg_models_all, gbl.h_t_c: hoexter_t_clf_models_all,
                   gbl.b_t_r: boedhoe_t_reg_models_all, gbl.b_t_c: boedhoe_t_clf_models_all}

    models_to_results(models2_all, subs.pat_frame_test_reg_norms, subs.pat_frame_test_clf_norms,
                      subs.pat_frame_test_y_reg, subs.pat_frame_test_y_clf, reg_scoring)

    # find best best brfc
    gbl.brfc_name = gbl.t_c + '_2_imb_train'
    brfc_scores = [[b[0].score(subs.pat_frame_test_reg_norms[n].loc[:, b[1]],
                               subs.pat_frame_test_y_clf), idx] for idx, b in enumerate(brfc_models_all)]
    brfc_scores.sort(key=lambda x: x[0], reverse=True)
    brfc_best5_models_with_feats = brfc_models_all[brfc_scores[0][1]]
    brfc_predictions = brfc_best5_models_with_feats[0].predict(
        subs.pat_frame_test_reg_norms[n].loc[:, brfc_best5_models_with_feats[1]])

    brfc_score = brfc_best5_models_with_feats[0].score(subs.pat_frame_test_reg_norms[n].loc[:, brfc_best5_models_with_feats[1]],
                                                       subs.pat_frame_test_y_clf)
    brfc_score2 = brfc_scores[0][0]
    if brfc_score != brfc_score2:
        print('BEST BRFC PRED SCORES NOT EQUAL')
    brfc_result = pd.DataFrame(index=subs.pat_frame_test_y_clf.index.tolist() + ['acc_score'],
                               data={gbl.brfc_name: brfc_predictions.tolist() + [brfc_score],
                                     'YBOCS_target': subs.pat_frame_test_y_clf.iloc[:, 0]})
    print(brfc_result)
    print(brfc_score)

    brfc_pr = brfc_result.drop(columns='YBOCS_target')

    # save best brfc feats
    gbl.feat_sets_best_train[gbl.brfc_name] = brfc_best5_models_with_feats[1]
    # construct manually brfc models to result
    brfc_bm = {'EstObject': brfc_best5_models_with_feats[0], 'est_type': 'brfc',
               'normIdx_train': n, 'num_feats': len(brfc_best5_models_with_feats[1])}
    gbl.best_models_results[gbl.brfc_name] = {'features': brfc_best5_models_with_feats[1],
                                              'est_class': 'clf',
                                              'best_model': brfc_bm,
                                              'pred_results': brfc_pr,
                                              'bm5': ''}

    # SAVE RESULTS
    print('SAVING RESULTS')
    # str(t_s) + \
    exp_description = '**balRandTest'+str(t_s)+'_RegTrainRest_ClfTrain' + over_samp_names[o_s] + '_' + norm + '_' \
                      + reg_scorers_names[r_sc] + '_' + clf_scorers_names[c_sc] + '_' + \
                      'cvFolds' + str(cv_folds) + \
                      '**t_allRegTrain_DesikanThickVolFeats_TorP'

    try:
        os.mkdir(clf_tgt)
    except FileExistsError:
        pass

    bmr = open(clf_tgt + '/' + clf_tgt + exp_description + '**bmr.pkl', 'wb')
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
    xlsx_name = clf_tgt + '/' + clf_tgt + exp_description + '**results**' + \
                'tclf:' +str(t_clf_best_score)+'_'+\
                'treg:' +str(t_reg_best_score)+'.xlsx'

    writer = pd.ExcelWriter(xlsx_name)
    write_report(writer, subs.pat_frame_test_y_clf, subs.pat_frame_test_y_reg)
    t_frame_name = 'vanilla' # SMOTE, ROS, ADASYN
    gbl.t_frame_global.to_excel(writer, 't_frame')

    writer.save()
    print(xlsx_name)

# end for clr_tgt
print("TOTAL TIME %.2f" % (time.time()-start_time))
