import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
import gbl
from fs_read import fs_data_collect
from estimators import regress, classify
from find_best_model import find_best_model
from scale import scale, testSet_scale
from prediction_reporting import predict_report, write_report
from pat_sets import *
from sklearn.model_selection import train_test_split
import time
import pickle
from pickling import *
from gdrive import get_pat_stats
import os
import warnings
from sklearn.exceptions import ConvergenceWarning
import copy
import sys
import random
from sklearn.feature_selection import VarianceThreshold
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.feature_selection import RFECV
from univariate import t_frame_compute
import xgboost
from sklearn.svm import SVC
from numpy.random import randint
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from models_to_results import models_to_results
from select_pat_names_test_clf import select_pat_names_test_clf

warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore" # Also affect subprocesses

# non-ocd (0-9) mild (10-20), moderate (21-30), severe (31-40) Okasha et. al. (2000)

start_time = time.time()

cv_folds = 10

# seed = 7
# np.random.seed(seed)

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

num_reg = len(gbl.regType_list)
num_clf = len(gbl.clfType_list)


reg_scorers = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_log_error', None]
reg_scorers_names = ['ev', 'nmae', 'nmsle', 'None']

r_sc = 1
reg_scoring = reg_scorers[r_sc]

clf_scorers = ['balanced_accuracy', 'accuracy']
clf_scorers_names = ['bac', 'ac']

c_sc = 0
clf_scoring = clf_scorers[c_sc]

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
# glob.demo_clin_feats = []

pat_frame = pd.concat([pat_frame, pat_frame_stats.loc[:, gbl.demo_clin_feats]], axis=1, sort=False)

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

    if 'obs' in clf_tgt:
        y_reg = pd.DataFrame({'obs': pat_frame_stats.loc[:, 'obs']})
    elif 'com' in clf_tgt:
        y_reg = pd.DataFrame({'com': pat_frame_stats.loc[:, 'com']})
    else:
        y_reg = pd.DataFrame({'YBOCS': pat_frame_stats.loc[:, 'YBOCS']})

    y_clf = pd.DataFrame({clf_tgt: pat_frame_stats.loc[:, clf_tgt]})
    num_classes = len(np.unique(y_clf))
    # extract train and test set names
    t_s = 0.19
    # pat_names_train, pat_names_test = train_test_split(pat_names,
    #                                                    test_size=t_s,
    #                                                    stratify=y_clf)
    #                                                    #random_state=random.randint(1, 101))

    pat_names_test = select_pat_names_test_clf(y_clf, clf_tgt, t_s, num_classes)
    pat_names_train = [name for name in pat_names if name not in pat_names_test]
    result = any(elem in pat_names_train for elem in pat_names_test)
    print(result)
    print('%d pat_names_test' % len(pat_names_test))
    print('%d pat_names_train' % len(pat_names_train))
    print(set(pat_names) == set(pat_names_test + pat_names_train))

    pat_frame_train = pat_frame.loc[pat_names_train, :]
    # create train set for reg
    pat_frame_train_reg = pat_frame_train
    pat_frame_train_reg_norms, pat_train_reg_scalers = scale(pat_frame_train_reg)

    pat_frame_train_y_reg = y_reg.loc[pat_names_train, :]

    # create train set for clf

    pat_frame_train_y_clf = y_clf.loc[pat_names_train, :]
    pat_frame_train_clf_list = [0,1,2,3]
    pat_frame_train_y_clf_list = [0,1,2,3]

    pat_frame_train_clf_list[0] = pat_frame_train
    pat_frame_train_y_clf_list[0] = pat_frame_train_y_clf

    over_samp_names = ['None', 'ROS', 'SMOTE', 'ADASYN']
    o_s = 0
    if num_classes <= 2:
        o_s = 0
    else:
        o_s = 3
        over_samp = [RandomOverSampler(random_state=random.randint(1, 101)),
                     SMOTE(random_state=random.randint(1,101)),
                     ADASYN(random_state=random.randint(1, 101))]

        for idx, over_sampler in enumerate(over_samp):
            try:
                a, b = over_sampler.fit_resample(pat_frame_train.values, pat_frame_train_y_clf.values)
                pat_frame_train_clf_list[idx+1] = pd.DataFrame(columns=pat_frame_train.columns, data=a)
                pat_frame_train_y_clf_list[idx+1] = pd.DataFrame(columns=pat_frame_train_y_clf.columns, data=b)
            except:
                pass

    pat_frame_train_clf_norms, pat_train_clf_scalers = scale(pat_frame_train_clf_list[o_s])
    pat_frame_train_y_clf = pat_frame_train_y_clf_list[o_s]

    # pat_test
    pat_frame_test = pat_frame.loc[pat_names_test, :]
    pat_frame_test_reg_norms = testSet_scale(pat_frame_test, pat_train_reg_scalers)
    pat_frame_test_clf_norms = testSet_scale(pat_frame_test, pat_train_clf_scalers)

    pat_frame_test_y_reg = y_reg.loc[pat_names_test, :]
    pat_frame_test_y_clf = y_clf.loc[pat_names_test, :]

    # con
    con_frame_norms, con_scalers = scale(con_frame)

    t_reg_models_all = []
    t_clf_models_all = []

    gbl.t_frame_perNorm_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

    # expert-picked-feature-based models for regression and classification
    hoexter_reg_models_all = []
    hoexter_clf_models_all = []
    boedhoe_reg_models_all = []
    boedhoe_clf_models_all = []

    brfc_models_all = []

    for n in iteration['n']: # standard, minmax, (robust-quantile-based) normalizations of input data
        t0 = time.time()
        norm = gbl.normType_list[n]

        # RFECV __________________________________
        # t_frame = t_compute(pat_frame_train, con_frame, n)
        #
        # # estimator = xgboost.XGBClassifier(random_state=randint(1, 101))
        # # estimator = SVC(C=500, kernel='linear',random_state=randint(1, 101))
        # # estimator = GradientBoostingClassifier(random_state=randint(1, 101))
        # estimator = AdaBoostClassifier(random_state=randint(1, 101))
        # selector = RFECV(estimator, min_features_to_select=10, cv=cv_folds, n_jobs=-1, step=1,
        #                  verbose=2, scoring='balanced_accuracy')
        #
        # selector.fit(pat_frame_train_clf_smote_resamp_norms[n],#.loc[:, t_frame.columns.tolist() + glob.demo_clin_feats],
        #              pat_frame_train_y_clf_smote_resamp)
        # predictions = selector.predict(pat_frame_test_clf_smote_resamp_norms[n])#.loc[:, t_frame.columns.tolist() + glob.demo_clin_feats])
        # reslt = pd.DataFrame(index=pat_frame_test_y_clf.index.tolist(),
        #                      data={'YBOCS_pred': predictions, 'YBOCS_target': pat_frame_test_y_clf.iloc[:, 0]})
        # print(reslt)
        # score = selector.score(pat_frame_test_clf_smote_resamp_norms[n],#.loc[:, t_frame.columns.tolist() + glob.demo_clin_feats],
        #                        pat_frame_test_y_clf)
        # print(score)
        # _______________________________________

        print("HOEXTER Regression with norm " + norm)

        # hoexter_reg_models_all += regress(pat_frame_train_reg_norms[n][glob.feat_sets_best_train[glob.h_r]],
        #                                   pat_frame_train_y_reg, cv_folds, reg_scoring, n, glob.h_r,
        #                                   len(glob.feat_sets_best_train[glob.h_r]))

        hoexter_clf_models_all += classify(pat_frame_train_clf_norms[n][gbl.feat_sets_best_train[gbl.h_c]],
                                           pat_frame_train_y_clf, cv_folds, clf_scoring, n, gbl.h_c,
                                           len(gbl.feat_sets_best_train[gbl.h_c]))
        # print("BOEDHOE Regression with norm " + norm)

        # boedhoe_reg_models_all += regress(pat_frame_train_reg_norms[n][glob.feat_sets_best_train[glob.b_r]],
        #                                   pat_frame_train_y_reg, cv_folds, reg_scoring, n, glob.b_r,
        #                                   len(glob.feat_sets_best_train[glob.b_r]))

        boedhoe_clf_models_all += classify(pat_frame_train_clf_norms[n][gbl.feat_sets_best_train[gbl.b_c]],
                                           pat_frame_train_y_clf, cv_folds, clf_scoring, n, gbl.b_c,
                                           len(gbl.feat_sets_best_train[gbl.b_c]))

        print("HOEXTER and BOEDHOE EST W/ NORM %s TOOK %.2f SEC" % (norm, time.time()-t0))

        # compute t_feats
        t_frame = t_frame_compute(pat_frame_train_reg, con_frame, n, ['thickness'])
        t_feats_all = t_frame.columns.tolist()
        t_feats_all_num = t_frame.shape[1]
        t_feats_train_all_num = t_feats_all_num + len(gbl.demo_clin_feats)
        print("FINISHED COMPUTING %d T VALUES" % t_feats_all_num)

        print(pd.DataFrame(data=t_feats_all))

        for idx2, t_feat in enumerate(t_feats_all):
            t0 = time.time()
            t_feats_num = idx2+1
            t_feats = t_feats_all[0:t_feats_num]

            t_feats_train = t_feats + gbl.demo_clin_feats
            t_feats_train_num = len(t_feats_train)

            print("COMPUTING %d / %d FEATS W/ NORM %s" % (t_feats_train_num,
                                                          t_feats_train_all_num,
                                                          norm))
            # t_reg_models_all += regress(pat_frame_train_reg_norms[n][t_feats_train],
            #                             pat_frame_train_y_reg, cv_folds,
            #                             reg_scoring, n, glob.t_r, t_feats_train_all_num)

            t_clf_models_all += classify(pat_frame_train_clf_norms[n][t_feats_train],
                                         pat_frame_train_y_clf, cv_folds,
                                         clf_scoring, n, gbl.t_c, t_feats_train_all_num)

            print("%s: Running brfc: %d OF %d FEATS" % (gbl.t_c, t_feats_train_num, t_feats_train_all_num))
            brfc_models_all.append([BalancedRandomForestClassifier(n_estimators=200,
                                                                   random_state=None,
                                                                   n_jobs=-1,
                                                                   class_weight='balanced').fit(pat_frame_train_reg_norms[n][t_feats_train],
                                                                                                pat_frame_train_y_clf_list[0]),
                                    t_feats_train])
            print("FINISHED %d / %d FEATS W/ NORM %s TOOK %.2f SEC" % (t_feats_train_num,
                                                                       t_feats_train_all_num,
                                                                       norm,
                                                                       time.time() - t0))

        # end for t_feats
    # end for n norm

    # find best trained models and prediction results
    models_all = {gbl.h_r: hoexter_reg_models_all, gbl.h_c: hoexter_clf_models_all,
                  gbl.b_r: boedhoe_reg_models_all, gbl.b_c: boedhoe_clf_models_all,
                  gbl.t_r: t_reg_models_all, gbl.t_c: t_clf_models_all}

    models_to_results(models_all, pat_frame_test_reg_norms, pat_frame_test_clf_norms,
                      pat_frame_test_y_reg, pat_frame_test_y_clf, reg_scoring)

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

    hoexter_t_clf_models_all = classify(pat_frame_train_clf_norms[n][gbl.feat_sets_best_train[gbl.h_t_c]],
                                        pat_frame_train_y_clf, cv_folds, clf_scoring, n, gbl.h_t_c,
                                        len(gbl.feat_sets_best_train[gbl.h_t_c]))
    # print("BOEDHOE Regression with norm " + norm)

    # boedhoe_t_reg_models_all = regress(pat_frame_train_reg_norms[n][glob.feat_sets_best_train[glob.b_t_r]],
    #                                    pat_frame_train_y_reg, cv_folds, reg_scoring, n, glob.b_t_r,
    #                                    len(glob.feat_sets_best_train[glob.b_t_r]))

    boedhoe_t_clf_models_all = classify(pat_frame_train_clf_norms[n][gbl.feat_sets_best_train[gbl.b_t_c]],
                                        pat_frame_train_y_clf, cv_folds, clf_scoring, n, gbl.b_t_c,
                                        len(gbl.feat_sets_best_train[gbl.b_t_c]))

    models2_all = {gbl.h_t_r: hoexter_t_reg_models_all, gbl.h_t_c: hoexter_t_clf_models_all,
                   gbl.b_t_r: boedhoe_t_reg_models_all, gbl.b_t_c: boedhoe_t_clf_models_all}

    models_to_results(models2_all, pat_frame_test_reg_norms, pat_frame_test_clf_norms,
                      pat_frame_test_y_reg, pat_frame_test_y_clf, reg_scoring)

    # find best best brfc
    gbl.brfc_name = gbl.t_c + '_2_imb_train'
    brfc_scores = [[b[0].score(pat_frame_test_reg_norms[n].loc[:, b[1]],
                               pat_frame_test_y_clf), idx] for idx, b in enumerate(brfc_models_all)]
    brfc_scores.sort(key=lambda x: x[0], reverse=True)
    brfc_best_model_with_feats = brfc_models_all[brfc_scores[0][1]]
    brfc_predictions = brfc_best_model_with_feats[0].predict(
        pat_frame_test_reg_norms[n].loc[:, brfc_best_model_with_feats[1]])

    brfc_score = brfc_best_model_with_feats[0].score(pat_frame_test_reg_norms[n].loc[:, brfc_best_model_with_feats[1]],
                                                     pat_frame_test_y_clf)
    brfc_score2 = brfc_scores[0][0]
    if brfc_score != brfc_score2:
        print('BEST BRFC PRED SCORES NOT EQUAL')
    brfc_result = pd.DataFrame(index=pat_frame_test_y_clf.index.tolist() + ['acc_score'],
                               data={gbl.brfc_name: brfc_predictions.tolist() + [brfc_score],
                                     'YBOCS_target': pat_frame_test_y_clf.iloc[:, 0]})
    print(brfc_result)
    print(brfc_score)

    brfc_pr = brfc_result.drop(columns='YBOCS_target')

    # save best brfc feats
    gbl.feat_sets_best_train[gbl.brfc_name] = brfc_best_model_with_feats[1]
    # construct manually brfc models to result
    brfc_bm = {'EstObject': brfc_best_model_with_feats[0], 'est_type': 'brfc',
               'normIdx_train': n, 'num_feats': len(brfc_best_model_with_feats[1])}
    gbl.best_models_results[gbl.brfc_name] = {'features': brfc_best_model_with_feats[1],
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
                      '**t_allRegTrain_onlyDesikanThickFeats_TorP'

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
    write_report(writer, pat_frame_test_y_clf, pat_frame_test_y_reg)
    for idx3, tfpNl in enumerate(gbl.t_frame_perNorm_list):
        tfpNl.to_excel(writer, 't_frame_' + gbl.normType_list[idx3])

    writer.save()
    print(xlsx_name)

# end for clr_tgt
print("TOTAL TIME %.2f" % (time.time()-start_time))
