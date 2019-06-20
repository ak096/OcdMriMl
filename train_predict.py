from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics.scorer import get_scorer
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVR, LinearSVC, SVC
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import sem, t
from scipy import mean

import gbl


def sort_criterion(elt):
    return elt['val_score']


def set_paramgrid_est(est_type, task):

    if est_type is gbl.linear_:
        if task is gbl.clf:
            param_grid = gbl.param_grid_lsvc
            est = LinearSVC()
            #est = SVC(kernel='linear', probability=True)
        elif task is gbl.reg:
            param_grid = gbl.param_grid_lsvr
            est = LinearSVR()
    elif est_type is gbl.non_linear_:
        param_grid = gbl.param_grid_xgb
        if task is gbl.clf:
            est = XGBClassifier()
        elif task is gbl.reg:
            est = XGBRegressor()
    return param_grid, est


def train(est_type, task, params, X, y, cv_folds, scoring=None):

    param_grid, est = set_paramgrid_est(est_type, task)

    est_gp_fits = []
    est5 = []
    val_scores5 = []
    print('training with cross_validation using scorer: %s' % scoring)

    # !!assumption val scores greater the better
    for idx, grid_point in enumerate(param_grid):
        grid_point.update(params)
        est.set_params(**grid_point)

        scores = cross_validate(est, X, y, groups=None, scoring=scoring, cv=cv_folds, n_jobs=-1, verbose=0,
                                fit_params=None, return_train_score=False, return_estimator=True,
                                error_score='raise-deprecating')
        # print('cv val scores for grid point are')
        # print(scores['test_score'])
        # print('choosing and storing best: index %d, score %.2f' % (np.argmax(scores['test_score']),
        #                                                            np.max(scores['test_score'])))

        est_gp_fits.append({'est': scores['estimator'][np.argmax(scores['test_score'])],
                            'val_score': np.max(scores['test_score'])
                           })

    est_gp_fits.sort(reverse=True, key=sort_criterion)
    # print('sorted best val scores over grid points')
    # for elt in est_gp_fits:
    #     print(elt['val_score'])
    print('choosing top 5 estimators per val scores')
    est_gp_fits_top5 = est_gp_fits[0:5]
    for egpf in est_gp_fits_top5:
        est5.append(egpf['est'])
        val_scores5.append(egpf['val_score'])

    return est5, val_scores5


def pred(est_type, task, est5, X, y, scoring=None):
    pred_frames = []
    pred_scores = []
    perm_imps = []
    print('prediction scores for best 5 estimators using scoring: %s' % scoring)
    for i, est in enumerate(est5):
        if scoring:
            scorer = get_scorer(scoring)
            ps = scorer(est, X, y)
        else:
            ps = est.score(X, y)

        pred_scores.append(ps)
        pred_frames.append(pd.DataFrame(index=y.index.tolist(),
                                        data={'YBOCS_pred': est.predict(X),
                                              'YBOCS_target': y}))
        # if task is gbl.clf:
        #     if est_type is gbl.linear_:
        #         pred_frames[i].insert(1, 'Confidence', est.decision_function(X))
        #     elif est_type is gbl.non_linear_:
        #         pred_frames[i].insert(1, 'Confidence', est.predict_proba(X))

        perm_imps.append(perm_imp_test(est=est, base_score=ps, X=X, y=y, n_iter=3, scoring=scoring))

    print(pred_scores)
    return pred_frames, pred_scores, perm_imps


def perm_imp_test(est, base_score, X, y, n_iter=3, scoring=None):
    perm_imp_frame = pd.DataFrame(index=['perm_imp'], columns=[c for c in X.columns.tolist() if c not in gbl.clin_demog_feats])

    for column in perm_imp_frame:

        X_col = deepcopy(X.loc[:, column])
        score_diff = 0.0
        for _ in np.arange(n_iter):
            X.loc[:, column] = np.random.permutation(X.loc[:, column])
            if scoring:
                scorer = get_scorer(scoring)
                score_diff += base_score - scorer(est, X, y)
            else:
                score_diff += base_score - est.score(X, y)
        X.loc[:, column] = X_col
        perm_imp_frame.at['perm_imp', column] = score_diff/n_iter
    return perm_imp_frame


# credit: kite
def conf_interval(data):
    confidence = 0.95
    n = len(data)
    m = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return [m, h]
