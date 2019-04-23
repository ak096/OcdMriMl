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
from pat_names_train_equal_per_class import *
from get_features import get_feats
from sklearn.model_selection import train_test_split
import time
import pickle
from pickling import load_data, save_data, remove_data
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

glob.init_globals()
cv_folds = 4

# seed = 7
# np.random.seed(seed)

# get data from FreeSurfer stats
path_base = os.path.abspath('Desktop/FS_SUBJ_ALL').replace('PycharmProjects/OcdMriMl/', '')

group = ['con', 'pat']

con_frame = fs_data_collect(group[0], path_base)
pat_frame = fs_data_collect(group[1], path_base)

pd.DataFrame({'con': con_frame.columns[con_frame.columns != pat_frame.columns], 
              'pat': pat_frame.columns[con_frame.columns != pat_frame.columns]})

glob.FS_feats = con_frame.columns.tolist()
num_pats = pat_frame.shape[0]
pat_names = pat_frame.index.tolist()
num_cons = con_frame.shape[0]

num_reg = len(glob.regType_list)
num_clr = len(glob.clrType_list)

reg_scoring = 'neg_mean_absolute_error'
clr_scoring = 'accuracy'

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
pat_frame = pd.concat([pat_frame, pat_frame_stats.loc[:, demo_clin_feats]], axis=1, sort=False)

load_data()

# test if load_data() worked
print(glob.iteration)
print("shape of hr_clr_models_all")
np.array(glob.hoexter_clr_models_all).shape

# load iteration loop values
clr_tgts = copy.deepcopy(glob.iteration['clr_targets'])
n = copy.deepcopy(glob.iteration['n'])
t_feats_num = copy.deepcopy(glob.iteration['t_feats_num'])


for clr_tgt in clr_tgts[:]:

    if 'obs' in clr_tgt:
        y_reg = pd.DataFrame({'obs': pat_frame_stats.loc[:, 'obs']})
    elif 'com' in clr_tgt:
        y_reg = pd.DataFrame({'com': pat_frame_stats.loc[:, 'com']})
    else:
        y_reg = pd.DataFrame({'YBOCS': pat_frame_stats.loc[:, 'YBOCS']})

    y_clr = pd.DataFrame({clr_tgt: pat_frame_stats.loc[:, clr_tgt]})

    # extract train and test sets
    pat_names_train, pat_names_test = train_test_split(pat_names, test_size=0.15, stratify=y_clr)


    pat_frame_train = pat_frame.loc[pat_names_train, :]
    pat_frame_y_train_reg = y_reg.loc[pat_names_train, :]
    pat_frame_y_train_clr = y_clr.loc[pat_names_train, :]

    pat_frame_test = pat_frame.loc[pat_names_test, :]
    pat_frame_y_test_reg = y_reg.loc[pat_names_test, :]
    pat_frame_y_test_clr = y_clr.loc[pat_names_test, :]

    pat_frame_train_norms, pat_frame_train_scalers = scale(pat_frame_train)
    pat_frame_test_norms = testSet_scale(pat_frame_test, pat_frame_train_scalers)
    con_frame_norms, con_frame_scalers = scale(con_frame)

    while True:  # mean, minmax, (robust-quantile-based) normalizations of input data
        t0 = time.time()
        norm = glob.normType_list[n]

        print("HOEXTER Regression with norm " + norm)
        hoexter_reg_models = regress(pat_frame_train_norms[n][hoexter_FSfeats + demo_clin_feats],
                                     pat_frame_y_train_reg, cv_folds, reg_scoring, n)
        print("HOEXTER Classification with norm " + norm)
        hoexter_clr_models = classify(pat_frame_train_norms[n][hoexter_FSfeats + demo_clin_feats],
                                      pat_frame_y_train_clr, cv_folds, clr_scoring, n)
        print("BOEDHOE Regression with norm " + norm)
        boedhoe_reg_models = regress(pat_frame_train_norms[n][boedhoe_FSfeats + demo_clin_feats],
                                     pat_frame_y_train_reg, cv_folds, reg_scoring, n)
        print("BOEDHOE Classification with norm " + norm)
        boedhoe_clr_models = classify(pat_frame_train_norms[n][boedhoe_FSfeats + demo_clin_feats],
                                      pat_frame_y_train_clr, cv_folds, clr_scoring, n)
        print()
        print("!!!!!!!!!!!!!!!!!!!!!!!!HOEXTER and BOEDHOE with norm %s took %.2f seconds" % (norm, time.time()-t0))
        print()

        print("COMPUTING T VALUES with norm " + norm)
        # t_test per feature
        t_frame = pd.DataFrame(index=['t_statistic', 'p_value'], columns=glob.FS_feats)
        for idx, feat in enumerate(glob.FS_feats):
            t_result = ttest_ind(pat_frame_train_norms[n].loc[:, feat], con_frame_norms[n].loc[:, feat])
            t_frame.iloc[0, idx] = t_result.statistic
            t_frame.iloc[1, idx] = t_result.pvalue

        t_frame.sort_values(by='t_statistic', axis=1, ascending=False, inplace=True)
        glob.t_frame_perNorm_list.append(t_frame)
        print('Dropping insignificant t features')
        for column in t_frame:
            if abs(t_frame.loc['t_statistic', column]) < 1.96 or abs(t_frame.loc['p_value', column]) > 0.05:
                t_frame.drop(columns=column, inplace=True)
        t_feats_num_total = t_frame.shape[1]
        print('%d t features left' % t_feats_num_total)

        t_feats_list = t_frame.columns.tolist()
        print("FINISHED COMPUTING T VALUES from norm " + norm)
        print()

        while True:  # through FS_t_feats
            t0 = time.time()
            t_feats = t_feats_list[0:t_feats_num]
            print(t_feats)
            pat_frame_train_t_feats = pat_frame_train_norms[n][t_feats + demo_clin_feats]

            print("Regression on %d tfeats with norm %s" % (t_feats_num, norm))
            t_reg_models = regress(pat_frame_train_t_feats, pat_frame_y_train_reg, cv_folds,
                                   reg_scoring, n)
            print("Classification on %d tfeats with norm %s" % (t_feats_num, norm))
            t_clr_models = classify(pat_frame_train_t_feats, pat_frame_y_train_clr, cv_folds,
                                    clr_scoring, n)
            print()
            print("!!!!!!!!!!!!!!!!!!!!!!!! %d t_feats with norm %s took %.2f seconds" % (t_feats_num, norm, time.time() - t0))
            print()
            save_data('t', t_reg_models, t_clr_models)

            glob.t_reg_models_all += t_reg_models
            glob.t_clr_models_all += t_clr_models

            t_reg_models = []
            t_clr_models = []

            t_feats_num += 1
            if t_feats_num > t_feats_num_total:
                t_feats_num = 1
                brk1 = True
            else:
                brk1 = False
            glob.iteration['t_feats_num'] = t_feats_num
            save_data('itr')
            if brk1:
                break
            # end while

        save_data('tfn')
        save_data('h', hoexter_reg_models, hoexter_clr_models)
        save_data('b', boedhoe_reg_models, boedhoe_clr_models)

        glob.hoexter_reg_models_all += hoexter_reg_models
        glob.hoexter_clr_models_all += hoexter_clr_models
        glob.boedhoe_reg_models_all += boedhoe_reg_models
        glob.boedhoe_clr_models_all += boedhoe_clr_models

        hoexter_reg_models = []
        hoexter_clr_models = []
        boedhoe_reg_models = []
        boedhoe_clr_models = []

        n += 1
        if n > 1:
            n = 0
            brk2 = True
        else:
            brk2 = False
        glob.iteration['n'] = n
        save_data('itr')
        if brk2:
            break
        # end while

    # find best trained models and prediction results
    best_models_results = {}
    models_all = {'hoexter_reg': glob.hoexter_reg_models_all, 'hoexter_clr': glob.hoexter_clr_models_all,
                  'boedhoe_reg': glob.boedhoe_reg_models_all, 'boedhoe_clr': glob.boedhoe_clr_models_all,
                  't_reg': glob.t_reg_models_all, 't_clr': glob.t_clr_models_all}

    for key, value in models_all.items():
        if value:
            bm = find_best_model(value)
            est_type = bm['est_type']
            pat_frame_test = pat_frame_test_norms[bm['normIdx_train']]
            ft = get_feats(key, bm) + demo_clin_feats
            if est_type in glob.regType_list:
                ec = 'reg'
                pr = predict_report(key, bm, pat_frame_test, ft, pat_frame_y_test_reg)
            elif est_type in glob.clrType_list:
                ec = 'clr'
                pr = predict_report(key, bm, pat_frame_test, ft, pat_frame_y_test_clr)

            best_models_results[key] = {'features': ft, 'est_class': ec, 'best_model': bm, 'pred_results': pr}

    # best_models_results[key] = {'features': [list],
    #                             'est_class': 'reg'||'clr',
    #                             'best_model': {'GridObject': , 'est_type': , 'normIdx_train': , 'num_feats': },
    #                             'pred_results': prediction_frame
    #                            }

    # SAVE RESULTS
    try:
        os.mkdir(clr_tgt)
    except FileExistsError:
        pass

    bmr = open(clr_tgt + '/' + clr_tgt + '_bmr.pkl', 'wb')
    pickle.dump(best_models_results, bmr, -1)
    bmr.close()

    # write prediction results to excel
    writer = pd.ExcelWriter(clr_tgt + '/' + clr_tgt + '_results.xlsx')
    write_report(writer, best_models_results, pat_frame_y_test_clr, pat_frame_y_test_reg)
    glob.t_frame_perNorm_list[0].to_excel(writer, 't_frame_' + glob.normType_list[0])
    glob.t_frame_perNorm_list[1].to_excel(writer, 't_frame_' + glob.normType_list[1])
    writer.save()

    glob.iteration['clr_targets'].remove(clr_tgt)
    remove_data()
    glob.init_globals(itr=False)
    save_data('itr')
    # end for loop
print("TOTAL TIME %.2f" % (time.time()-start_time))
