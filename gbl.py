import os

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid, ParameterSampler
from sklearn.feature_selection import VarianceThreshold

from FreeSurfer_read import FreeSurfer_data_collect
from gdrive import get_pat_stats

pd.options.mode.use_inf_as_na = True


def svm_hyper_param_space(est_class):
    svm_hps = {
               'C': np.arange(1, 1001, 99),
               }
    if est_class == 'reg':
        svm_hps['epsilon'] = [0.3, 0.5, 0.7, 0.9]
    if est_class == 'clf':
        svm_hps['class_weight'] = ['balanced']
    return svm_hps


def xgb_hyper_param_space():
    return {  #learning task parameters : objective, eval, ...
              #'objective': ,
              #'eval_meas': ,
              #booster tree parameters
              #1.set init values, comp less expensive for initial look
              #for highly imbalanced classes
              'scale_pos_weight': [1],
              #2.most impact these two
              'min_child_weight': [1, 2, 3, 4, 5, 6],  #default:1
              "max_depth": [3, 4, 5, 6, 7, 8], # default:6
              #3.carry on with gamma
              "min_split_loss": [0, 0.1, 0.2, 0.3],  # alias:gamma, default:0 should be tuned according to loss function
              #4.these two around 0.8
              "subsample": [0.5, 0.6, 0.7, 0.8, 0.9],  # default
              'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9],
              #5.regularization parameters : model complexity and performance and under/over fitting?
              'reg_lambda': [1], #alias:lambda (L2), default:1
              'reg_alpha': [0, 0.001, 0.005, 0.01, 0.05], #alias:alpha (L1), default:0
              #6.decrease learning rate and increase number of trees
              "learning_rate": [0.1, 0.2],  # alias:eta, default:0.3
              'n_estimators': np.arange(100, 701, 100),  # default:100
            }


def _add_rand_feats(frame):
    # add some rand feat (randomly permuted features) to original data frame before train/test

    rand_feats = list(np.random.choice(frame.columns.tolist(), n_rand_feat, replace=False))
    for i in np.arange(n_rand_feat):
        rand_feat_name = 'RANDOM_' + str(i) + '_' + rand_feats[i]
        frame[rand_feat_name] = np.random.permutation(frame.loc[:, rand_feats[i]])
    return frame
# def init_globals():


normType_list = ['std', 'minMax', 'robust']

clin_demog_feats = ['gender_num', 'age', 'duration', 'med']

linear_ = 'linear'
non_linear_ = 'non_linear'

clf = 'clf'
reg = 'reg'

grid_space_size = 25

param_grid_lsvc = list(ParameterSampler(svm_hyper_param_space('clf'), n_iter=grid_space_size))

param_grid_lsvr = list(ParameterSampler(svm_hyper_param_space('reg'), n_iter=grid_space_size))

param_grid_xgb = list(ParameterSampler(xgb_hyper_param_space(), n_iter=grid_space_size))
# param_grid_xgbr = list(ParameterSampler(xgb_hyper_param_space(), n_iter=20))

# Hoexter et al 2013 (CSTC)
# volumetric data:
# right rostral anteriorcingulate
# left rostral anteriorcingulate
# right thalamus
# left thalamus
# right medial orbitofrontal
# right lateral orbitofrontal
# left medial orbitofrontal
# left lateral orbitofrontal
# right accumbens area (?)
# right pallidum
# right putamen
# right caudate
# left accumbens area (?)
# left pallidum
# left putamen
# left caudate

hoexter_feats_Desikan = [
                        'lh_rostralanteriorcingulate_volume**aparc',
                        'rh_rostralanteriorcingulate_volume**aparc',

                        'Right-Thalamus-Proper_volume',
                        'Left-Thalamus-Proper_volume',

                        'rh_medialorbitofrontal_volume**aparc',
                        'lh_medialorbitofrontal_volume**aparc',

                        'rh_lateralorbitofrontal_volume**aparc',
                        'lh_lateralorbitofrontal_volume**aparc',

                        'Left-Accumbens-area',
                        'Right-Accumbens-area',

                        'Left-Pallidum_volume',
                        'Right-Pallidum_volume',

                        'Right-Putamen_volume',
                        'Left-Putamen_volume',

                        'Left-Caudate_volume',
                        'Right-Caudate_volume'
                        ]

# Boedhoe et al 2016 (Pallidum, Hippocampus)

boedhoe_feats_Desikan = [
                        'Left-Pallidum_volume',
                        'Right-Pallidum_volume',
                        'Left-Hippocampus_volume',
                        'Right-Hippocampus_volume'
                        ]

# get data from FreeSurfer stats
path_base = os.path.abspath('Desktop/FS_SUBJ_ALL').replace('PycharmProjects/OcdMriMl/', '')
print('PreProc: FreeSurfer read: pat and con')
pat_frame = FreeSurfer_data_collect('pat', path_base)
con_frame = FreeSurfer_data_collect('con', path_base)

if not pat_frame.columns.tolist() == con_frame.columns.tolist():
    print('PreProc: FreeSurfer read: pat and con frame not equal!')
    exit()

# remove low variance features
before = pat_frame.shape[1]
threshold = 0.01
sel = VarianceThreshold(threshold=threshold)
sel.fit(pat_frame)
retained_mask = sel.get_support(indices=False)
pat_frame = pat_frame.loc[:, retained_mask]
after = pat_frame.shape[1]
print('PreProc: %d feats. removed: under %.2f var: %d to %d' % (before - after, threshold, before, after))

# remove features less than 90% populated
before = pat_frame.shape[1]
ratio = 1.00
threshold = round(ratio * pat_frame.shape[0])
pat_frame.dropna(axis='columns', thresh=threshold, inplace=True)
after = pat_frame.shape[1]
print('PreProc: %d feats. removed: less than %.2f percent filled: %d to %d' %
      (before - after, (threshold / pat_frame.shape[0]) * 100, before, after))

# add some random features (permuted original features)
before = pat_frame.shape[1]
n_rand_feat = 3
pat_frame = _add_rand_feats(pat_frame)
after = pat_frame.shape[1]
print('PreProc: %d feats. (rand.) added: %d to %d' % (n_rand_feat, before, after))

FreeSurfer_feats = pat_frame.columns.tolist()

# add clin demo features
pat_frame_stats = get_pat_stats()
pat_frame.sort_index(inplace=True)
pat_frame_stats.sort_index(inplace=True)
if pat_frame_stats.index.tolist() != pat_frame.index.tolist():
    exit("PreProc: feature and target pats not same!")
pat_frame = pd.concat([pat_frame, pat_frame_stats.loc[:, clin_demog_feats]], axis=1, sort=False)


