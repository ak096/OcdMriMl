import pandas as pd


def init_globals(frame):
    global regType_list
    regType_list = ['rfr', 'svmr', 'mlpr', 'lr', 'enr', 'rr', 'lasr', 'laslarr', 'gbr', 'xgbr']
    global clfType_list
    clfType_list = ['rfc', 'svmc', 'mlpc', 'abc', 'logr', 'knc', 'gpc', 'gnb', 'lda', 'qda', 'gbc', 'xgbc', 'brfc']
    global normType_list
    normType_list = ['std', 'minMax', 'robust']
    global FS_feats
    FS_feats = frame.columns.tolist()
    global t_frame_perNorm_list
    t_frame_perNorm_list = []

    global demo_clin_feats
    demo_clin_feats = ['gender_num', 'age', 'duration', 'med']

    # for get_features mostly
    global h_r
    h_r = 'hoexter_reg'
    global h_c
    h_c = 'hoexter_clf'
    global b_r
    b_r = 'boedhoe_reg'
    global b_c
    b_c = 'boedhoe_clf'
    global t_r
    t_r = 't_reg'
    global t_c
    t_c = 't_clf'
    global h_t_r
    h_t_r = 'hoexter_t_reg'
    global h_t_c
    h_t_c = 'hoexter_t_clf'
    global b_t_r
    b_t_r = 'boedhoe_t_reg'
    global b_t_c
    b_t_c = 'boedhoe_t_clf'
    global brfc_name
    brfc_name = t_c + '_2_imb_train'

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
    global hoexter_feats_FS
    hoexter_feats_FS = [
        'lh_rostralanteriorcingulate_volume**aparc',
        'rh_rostralanteriorcingulate_volume**aparc',

        'Right-Thalamus-Proper**volume',
        'Left-Thalamus-Proper**volume',

        'rh_medialorbitofrontal_volume**aparc',
        'lh_medialorbitofrontal_volume**aparc',

        'rh_lateralorbitofrontal_volume**aparc',
        'lh_lateralorbitofrontal_volume**aparc',

        'Left-Accumbens-area**volume',
        'Right-Accumbens-area**volume',

        'Left-Pallidum**volume',
        'Right-Pallidum**volume',

        'Right-Putamen**volume',
        'Left-Putamen**volume',

        'Left-Caudate**volume',
        'Right-Caudate**volume'
    ]

    # Boedhoe et al 2016 (Pallidum, Hippocampus)
    global boedhoe_feats_FS
    boedhoe_feats_FS = [
        'Left-Pallidum**volume',
        'Right-Pallidum**volume',
        'Left-Hippocampus**volume',
        'Right-Hippocampus**volume'
    ]

    global feat_sets_best_train
    feat_sets_best_train = {h_r: [], h_c: [], b_r: [], b_c: [], t_r: [], t_c: [],
                            h_t_r: [], h_t_c: [], b_t_r: [], b_t_c: []}

    # best_models_results[key] = {'features': [list],
    #                             'est_class': 'reg'||'clf',
    #                             'best_model': {'EstObject': , 'est_type': , 'normIdx_train': , 'num_feats': },
    #                             'pred_results': prediction_frame
    #                             'bm5': top 5 best scoring models for confidence interval comparison
    #                            }
    global best_models_results
    best_models_results = {}

    return
