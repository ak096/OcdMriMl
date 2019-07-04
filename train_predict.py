from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.metrics.scorer import get_scorer
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVR, LinearSVC, SVC
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from scipy.stats import sem, t
from scipy import mean

import gbl


def set_paramgrid_est(est_type, task):

    if est_type is gbl.linear_:
        if task is gbl.clf:
            param_grid = gbl.lsvc_hypparam_grid
            est = LinearSVC()
            #est = SVC(kernel='linear', probability=True)
        elif task is gbl.reg:
            param_grid = gbl.lsvr_hypparam_grid
            est = LinearSVR()
    elif est_type is gbl.non_linear_:
        param_grid = gbl.gbe_hypparam_grid #gbl.param_grid_xgb
        if task is gbl.clf:
            est = GradientBoostingClassifier() #XGBClassifier()
        elif task is gbl.reg:
            est = GradientBoostingRegressor() #XGBRegressor()
    return param_grid, est


def train(est_type, task, params, X, y, cv_folds, scoring=None, thresh=None):

    param_grid, est = set_paramgrid_est(est_type, task)

    ests_gpfits = []

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

        ests_gpfits += [(scores['estimator'][idx], round(ts, 3)) for idx, ts in
                        enumerate(scores['test_score']) if ts > thresh]
        # ests_gpfits += (scores['estimator'][np.argmax(scores['test_score'])],
        #                  round(np.max(scores['test_score']), 3))
    ests, train_scores = zip(*ests_gpfits)

    return list(ests), list(train_scores)


def pred(est_type, task, ests, X, y, scoring=None, thresh=None):
    pred_frames = []
    pred_scores = []
    perm_imps = []

    for est in ests:
        if scoring:
            scorer = get_scorer(scoring)
            ps = scorer(est, X, y)
        else:
            ps = est.score(X, y)

        pred_scores.append(round(ps, 3))
        pred_frames.append(pd.DataFrame(index=y.index.tolist(),
                                        data={'YBOCS_pred': est.predict(X),
                                              'YBOCS_target': y}))
        if ps > thresh:
            perm_imp_test(task, est, ps, X, y, 1, scoring)
        # if task is gbl.clf:
        #     if est_type is gbl.linear_:
        #         pred_frames[i].insert(1, 'Confidence', est.decision_function(X))
        #     elif est_type is gbl.non_linear_:
        #         pred_frames[i].insert(1, 'Confidence', est.predict_proba(X))

        #perm_imps.append(perm_imp_test(est=est, base_score=ps, X=X, y=y, n_iter=3, scoring=scoring))

    return pred_scores, pred_frames#, perm_imps


def perm_imp_test(task, est, base_score, X, y, n_iter=1, scoring=None):
    feats = [c for c in X.columns.tolist() if c not in gbl.clin_demog_feats]
    for f in feats:
        X_col = deepcopy(X.loc[:, f])
        score_diff = 0.0
        for _ in np.arange(n_iter):
            X.loc[:, f] = np.random.permutation(X.loc[:, f])
            if scoring:
                scorer = get_scorer(scoring)
                score_diff += base_score - scorer(est, X, y)
            else:
                score_diff += base_score - est.score(X, y)
            X.loc[:, f] = X_col
        if task is gbl.clf:
            gbl.fpis_clf.setdefault(f, []).append(score_diff/n_iter)
        elif task is gbl.reg:
            gbl.fpis_reg.setdefault(f, []).append(score_diff/n_iter)

    return


# credit: kite
def conf_interval(data):
    confidence = 0.95
    n = len(data)
    m = mean(data)
    std_err = sem(data)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    return [m, h]
