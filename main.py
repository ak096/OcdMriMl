import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from config import *
from fs_read import fs_data_collect
from estimators import regress, classify
from find_best_model import find_best_model
from scale import scale, testSet_scale
from prediction_reporting import predict_report, write_report
from hoexter_features import hoexter_FSfeats
from boedhoe_features import boedhoe_FSfeats
from get_features import get_feats
from sklearn.metrics import mean_absolute_error, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import time
import pickle
from pickling import load_data, save_data

start_time = time.time()
seed = 1
np.random.seed(seed)

# get data from FreeSurfer stats
path_base = '/home/bran/Desktop/FS_SUBJ_ALL/'

group = ['con', 'pat']

con_frame = fs_data_collect(group[0], path_base)
pat_frame = fs_data_collect(group[1], path_base)

pd.DataFrame({'con': con_frame.columns[con_frame.columns != pat_frame.columns], 
              'pat': pat_frame.columns[con_frame.columns != pat_frame.columns]})

FS_feats = con_frame.columns
num_pats = pat_frame.shape[0]
pat_names = pat_frame.index.values.tolist()
num_cons = con_frame.shape[0]

num_reg = len(regType_list)
num_clr = len(clrType_list)

# get patient stats, labels, ybocs ...
y = open('/home/bran/Desktop/data_master_akshay/akshay_pat_stats.csv')
pat_stats = pd.DataFrame()
pat_stats = pd.read_csv(y, index_col=0)
y.close()
if pat_stats.index.all() == pat_frame.index.all():
    print("FS pats and YBOCS pats indices are same")

ybocs = pd.DataFrame({'YBOCS_total': pat_stats.loc[:, 'YBOCS_total']})

# non-ocd (0-9) mild (10-20), moderate (21-30), severe (31-40) Okasha et. al. (2000)
ybocs_classes = pd.DataFrame({'YBOCS_class_4': pat_stats.loc[:, 'YBOCS_class_4']})

# extract train and test sets
pat_train, pat_test = train_test_split(pat_names, test_size=0.1)

pat_frame_train = pat_frame.loc[pat_train]
pat_frame_y_train_reg = ybocs.loc[pat_train]
pat_frame_y_train_clr = ybocs_classes.loc[pat_train]

pat_frame_test = pat_frame.loc[pat_test]
pat_frame_y_test_reg = ybocs.loc[pat_test]
pat_frame_y_test_clr = ybocs_classes.loc[pat_test]

pat_frame_train_norms, pat_frame_train_scalers = scale(pat_frame_train)
pat_frame_test_norms = testSet_scale(pat_frame_test, pat_frame_train_scalers)
con_frame_norms, con_frame_scalers = scale(con_frame)

cv_folds = 3

reg_scoring = 'neg_mean_absolute_error'
clr_scoring = 'accuracy'

hoexter_reg_models = []
hoexter_clr_models = []
boedhoe_reg_models = []
boedhoe_clr_models = []
t_reg_models = []
t_clr_models = []

load_data()
n = iteration['n']
num_tfeats = iteration['num_tfeats']


while True:  # mean, minmax, (robust-quantile-based) normalizations of input data
    t0 = time.time()
    norm = normType_list[n]

    print("HOEXTER Regression with norm " + norm)
    hoexter_reg_models = regress(pat_frame_train_norms[n][hoexter_FSfeats], pat_frame_y_train_reg, cv_folds,
                                     reg_scoring, normType_list[n])
    print("HOEXTER Classification with norm " + norm)
    hoexter_clr_models = classify(pat_frame_train_norms[n][hoexter_FSfeats], pat_frame_y_train_clr, cv_folds,
                                       clr_scoring, norm)
    print("BOEDHOE Regression with norm " + norm)
    boedhoe_reg_models = regress(pat_frame_train_norms[n][boedhoe_FSfeats], pat_frame_y_train_reg, cv_folds,
                                      reg_scoring, normType_list[n])
    print("BOEDHOE Classification with norm " + norm)
    boedhoe_clr_models = classify(pat_frame_train_norms[n][boedhoe_FSfeats], pat_frame_y_train_clr, cv_folds,
                                       clr_scoring, normType_list[n])
    print()
    print("!!!!!!!!!!!!!!!!!!!!!!!!HOEXTER and BOEDHOE with norm %s took %.2f seconds" % (norm, time.time()-t0))
    print()

    print("COMPUTING T VALUES with norm " + norm)
    # t_test per feature
    t_frame = pd.DataFrame(index=['t_statistic', 'p_value'], columns=FS_feats)
    for idx, feat in enumerate(FS_feats):
        t_result = ttest_ind(pat_frame_train_norms[n].loc[:, feat], con_frame_norms[n].loc[:, feat])
        t_frame.iloc[0, idx] = t_result.statistic
        t_frame.iloc[1, idx] = t_result.pvalue

    t_frame.sort_values(by='t_statistic', axis=1, ascending=False, inplace=True)
    t_frame_perNorm_list.append(t_frame)

    t_feats = t_frame.columns.tolist()
    print("FINISHED COMPUTING T VALUES with norm " + norm)
    print()
    while True:  # through FS_tfeats
        t0 = time.time()
        t_feats = t_feats[0:num_tfeats]

        pat_frame_train_feats = pat_frame_train_norms[n][t_feats]
        print("Regression on %d tfeats with norm %s" % (num_tfeats, norm))
        t_reg_models = regress(pat_frame_train_feats, pat_frame_y_train_reg, cv_folds,
                                    reg_scoring, normType_list[n])
        print("Classification on %d tfeats with norm %s" % (num_tfeats, norm))
        t_clr_models = classify(pat_frame_train_feats, pat_frame_y_train_clr, cv_folds,
                                     clr_scoring, normType_list[n])
        print()
        print("!!!!!!!!!!!!!!!!!!!!!!!! %d t_feats with norm %s took %.2f seconds" % (num_tfeats, norm, time.time()-t0))
        print()
        save_data('t', t_reg_models, t_clr_models)

        t_reg_models_all += t_reg_models
        t_clr_models_all += t_clr_models

        t_reg_models = []
        t_clr_models = []

        num_tfeats += 1
        if num_tfeats > len(FS_feats):
            iteration['num_tfeats'] = 1
            brk = True
        else:
            iteration['num_feats'] = num_tfeats
            brk = False
        save_data('itr')
        if brk:
            break

    # end while

    save_data('h', hoexter_reg_models, hoexter_clr_models)
    save_data('b', boedhoe_reg_models, boedhoe_clr_models)

    hoexter_reg_models_all += hoexter_reg_models
    hoexter_clr_models_all += hoexter_clr_models
    boedhoe_reg_models_all += boedhoe_reg_models
    boedhoe_clr_models_all += boedhoe_clr_models

    hoexter_reg_models = []
    hoexter_clr_models = []
    boedhoe_reg_models = []
    boedhoe_clr_models = []

    n += 1
    iteration['n'] = n
    save_data('itr')
    if n > 2:
        break

# end while

# find best trained models and prediction results
best_models_results = {}
models_all = {'hoexter_reg': hoexter_reg_models_all, 'hoexter_clr': hoexter_clr_models_all,
              'boedhoe_reg': boedhoe_reg_models_all, 'boedhoe_clr': boedhoe_clr_models_all,
              't_reg': t_reg_models_all, 't_clr': t_clr_models_all}

for key, value in models_all.items():
    if value:
        bm = find_best_model(value)
        est_type = bm['est_type']
        pat_frame_test = pat_frame_test_norms[normType_list.index(bm['normType_train'])]
        ft = get_feats(key, bm)
        if est_type in regType_list:
            ec = 'reg'
            pr = predict_report(key, bm, pat_frame_test, ft, pat_frame_y_test_reg)
        elif est_type in clrType_list:
            ec = 'clr'
            pr = predict_report(key, bm, pat_frame_test, ft, pat_frame_y_test_clr)

        best_models_results[key] = {'features': ft, 'est_class': ec, 'best_model': bm, 'pred_results': pr}

bmr = open('bmr.pkl', 'wb')
pickle.dump(best_models_results, bmr, -1)
bmr.close()

# best_models_results[key] = {'features': [list],
#                             'est_class': {'reg'|'clr},
#                             'best_model': {'GridObject': , 'est_type': , 'normType_train': , 'num_feats': },
#                             'pred_results': prediction_frame
#                             }

# write prediction results to excel
writer = pd.ExcelWriter('results.xlsx')

write_report(writer, best_models_results, pat_frame_y_test_clr, pat_frame_y_test_reg)

writer.save()
print("TOTAL TIME %.2f days" % (time.time()-start_time)*1.15741e-5)
