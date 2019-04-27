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

glob.init_globals(con_frame)
print('%d FS_FEATS READ' % len(glob.FS_feats))

num_pats = pat_frame.shape[0]
pat_names = pat_frame.index.tolist()
num_cons = con_frame.shape[0]

num_reg = len(glob.regType_list)
num_clr = len(glob.clrType_list)

reg_scoring = 'neg_mean_absolute_error' #'‘explained_variance’
clr_scoring = 'balanced_accuracy'
#clr_scoring = 'accuracy'

# get patient stats, labels, ybocs ...
scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('ocdmriml_gdrive_client_secret.json', scope)
gsclient = gspread.authorize(creds)
pat_stats_sheet = gsclient.open("AKSHAY_pat_stats").sheet1

pat_frame_stats = pd.DataFrame(pat_stats_sheet.get_all_records())
pat_frame_stats.index = pat_frame_stats.loc[:, 'subject']
pat_frame_stats.drop(columns=['', 'subject'])

if True in pat_frame_stats.index != pat_frame.index:
    exit("feature and target pats not same")

demo_clin_feats = ['gender_num', 'age', 'duration', 'med']
# demo_clin_feats = []

pat_frame = pd.concat([pat_frame, pat_frame_stats.loc[:, demo_clin_feats]], axis=1, sort=False)

iteration = {'n': [0],
             'clr_targets': [#'obs_class3_scorerange',
                             #'com_class3_scorerange',
                             'YBOCS_class3_scorerange']
                             #'YBOCS_class4_scorerange']
                             #'obs_class3_equalpat',
                             #'com_class3_equalpat',
                             #'YBOCS_class3_equalpat']
             }


for clr_tgt in iteration['clr_targets']:

    if 'obs' in clr_tgt:
        y_reg = pd.DataFrame({'obs': pat_frame_stats.loc[:, 'obs']})
    elif 'com' in clr_tgt:
        y_reg = pd.DataFrame({'com': pat_frame_stats.loc[:, 'com']})
    else:
        y_reg = pd.DataFrame({'YBOCS': pat_frame_stats.loc[:, 'YBOCS']})

    y_clr = pd.DataFrame({clr_tgt: pat_frame_stats.loc[:, clr_tgt]})

    # extract train and test sets
    t_s = 0.15
    pat_names_train, pat_names_test = train_test_split(pat_names, test_size=t_s, stratify=y_clr)
    #pat_names_train = pat_names_train_equal_per_YBOCS_class3
    #pat_names_test = pat_names_test_equal_per_YBOCS_class3

    pat_frame_train = pat_frame.loc[pat_names_train, :]
    pat_frame_y_train_reg = y_reg.loc[pat_names_train, :]
    pat_frame_y_train_clr = y_clr.loc[pat_names_train, :]

    pat_frame_test = pat_frame.loc[pat_names_test, :]
    pat_frame_y_test_reg = y_reg.loc[pat_names_test, :]
    pat_frame_y_test_clr = y_clr.loc[pat_names_test, :]

    pat_frame_train_norms, pat_frame_train_scalers = scale(pat_frame_train)
    pat_frame_test_norms = testSet_scale(pat_frame_test, pat_frame_train_scalers)
    con_frame_norms, con_frame_scalers = scale(con_frame)

    t_reg_models_all = []
    t_clr_models_all = []

    glob.t_frame_perNorm_list = [pd.DataFrame(), pd.DataFrame(), pd.DataFrame()]

    # expert-picked-feature-based models for regression and classification
    hoexter_reg_models_all = []
    hoexter_clr_models_all = []
    boedhoe_reg_models_all = []
    boedhoe_clr_models_all = []

    for n in iteration['n']: # standard, minmax, (robust-quantile-based) normalizations of input data
        t0 = time.time()
        norm = glob.normType_list[n]

        # print("HOEXTER Regression with norm " + norm)
        hoexter_reg_models_all += regress(pat_frame_train_norms[n][hoexter_FSfeats + demo_clin_feats],
                                          pat_frame_y_train_reg, cv_folds, reg_scoring, n)

        hoexter_clr_models_all += classify(pat_frame_train_norms[n][hoexter_FSfeats + demo_clin_feats],
                                           pat_frame_y_train_clr, cv_folds, clr_scoring, n)
        # print("BOEDHOE Regression with norm " + norm)
        boedhoe_reg_models_all += regress(pat_frame_train_norms[n][boedhoe_FSfeats + demo_clin_feats],
                                          pat_frame_y_train_reg, cv_folds, reg_scoring, n)

        boedhoe_clr_models_all += classify(pat_frame_train[boedhoe_FSfeats + demo_clin_feats],
                                           pat_frame_y_train_clr, cv_folds, clr_scoring, n)

        print("HOEXTER and BOEDHOE EST W/ NORM %s TOOK %.2f SEC" % (norm, time.time()-t0))

        # t_test per feature
        t_frame = pd.DataFrame(index=['t_statistic', 'p_value'], columns=glob.FS_feats)
        print(t_frame)
        for feat in glob.FS_feats:
            t_result = ttest_ind(pat_frame_train.loc[:, feat],
                                 con_frame.loc[:, feat])
            print(t_result)
            t_frame.at['t_statistic', feat] = t_result.statistic
            t_frame.at['p_value', feat] = t_result.pvalue
            print('%s t:%f p:%f' % (feat, t_frame.loc['t_statistic', feat], t_frame.loc['p_value', feat]))
        t_frame.sort_values(by='t_statistic', axis=1, ascending=False, inplace=True)
        glob.t_frame_perNorm_list[n] = t_frame

        for column in t_frame:
            if abs(t_frame.loc['t_statistic', column]) <= 1.96 or abs(t_frame.loc['p_value', column]) >= 0.05:
                t_frame.drop(columns=column, inplace=True)
                print('dropping %s' % column)

        t_feats_num_total = t_frame.shape[1]

        t_feats_list = t_frame.columns.tolist()
        print("FINISHED COMPUTING %d T VALUES" % t_feats_num_total)

        print(t_feats_list)

        for idx2, t_feat in enumerate(t_feats_list):
            t0 = time.time()
            t_feats_num = idx2+1
            t_feats = t_feats_list[0:t_feats_num]

            pat_frame_train_t_feats = pat_frame_train_norms[n][t_feats + demo_clin_feats]

            t_reg_models_all += regress(pat_frame_train_t_feats, pat_frame_y_train_reg, cv_folds,
                                        reg_scoring, n)

            t_clr_models_all += classify(pat_frame_train_t_feats, pat_frame_y_train_clr, cv_folds,
                                         clr_scoring, n)

            print("%d / %d T FEATS EST W/ NORM %s TOOK %.2f SEC" % (t_feats_num,
                                                                    t_feats_num_total,
                                                                    norm,
                                                                    time.time() - t0))
            # end for t_feats
        # end for n
    # end for clr_targets

    # find best trained models and prediction results
    best_models_results = {}
    models_all = {'hoexter_reg': hoexter_reg_models_all, 'hoexter_clr': hoexter_clr_models_all,
                  'boedhoe_reg': boedhoe_reg_models_all, 'boedhoe_clr': boedhoe_clr_models_all,
                  't_reg': t_reg_models_all, 't_clr': t_clr_models_all}

    for key, value in models_all.items():
        if value:
            bm = find_best_model(key, value)
            est_type = bm['est_type']
            pat_frame_test = pat_frame_test_norms[bm['normIdx_train']]
            ft = get_feats(key, bm) + demo_clin_feats
            if est_type in glob.regType_list:
                ec = 'reg'
                pr = predict_report(key, bm, pat_frame_test, ft, pat_frame_y_test_reg, ec)
            elif est_type in glob.clrType_list:
                ec = 'clr'
                pr = predict_report(key, bm, pat_frame_test, ft, pat_frame_y_test_clr, ec)

            best_models_results[key] = {'features': ft, 'est_class': ec, 'best_model': bm, 'pred_results': pr}

    # best_models_results[key] = {'features': [list],
    #                             'est_class': 'reg'||'clr',
    #                             'best_model': {'GridObject': , 'est_type': , 'normIdx_train': , 'num_feats': },
    #                             'pred_results': prediction_frame
    #                            }

    # SAVE RESULTS
    print('SAVING RESULTS')
    exp_description = '**skStrat_test'+str(t_s)+'_'+norm+'_balAcc_nmae_cvFolds'+str(cv_folds)+\
                      '**t_allTrain'

    # try:
    #     os.mkdir(clr_tgt + exp_description)
    # except FileExistsError:
    #     pass

    # bmr = open(clr_tgt + exp_description + '/' + clr_tgt + exp_description + '_bmr.pkl', 'wb')
    # pickle.dump(best_models_results, bmr, -1)
    # bmr.close()

    # write prediction results to excel
    # xlsx_name = clr_tgt + exp_description + '/' + clr_tgt + exp_description + '_results.xlsx'
    xlsx_name = clr_tgt + exp_description + '**results.xlsx'

    writer = pd.ExcelWriter(xlsx_name)
    write_report(writer, best_models_results, pat_frame_y_test_clr, pat_frame_y_test_reg)
    for idx3, tfpNl in enumerate(glob.t_frame_perNorm_list):
        tfpNl.to_excel(writer, 't_frame_' + glob.normType_list[idx3])

    writer.save()
    print(xlsx_name)

    # end for loop
print("TOTAL TIME %.2f" % (time.time()-start_time))
