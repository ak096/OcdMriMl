import gbl
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVR, LinearSVC
from xgboost import XGBRegressor, XGBClassifier
from copy import deepcopy
import numpy as np
import pandas as pd


def sort_criterion(elt):
    return elt['score']


def train(est_type, task, params, X, y, cv_folds, scoring=None):

    if est_type is 'linear':
        if task is 'clf':
            param_grid = gbl.param_grid_lsvc
            est = LinearSVC()
        elif task is 'reg':
            param_grid = gbl.param_grid_lsvr
            est = LinearSVR()
    elif est_type is 'non-linear':
        param_grid = gbl.param_grid_xgb
        if task is 'clf':
            est = XGBClassifier()
        elif task is 'reg':
            est = XGBRegressor()

    est_gp_fits = []
    est5 = []
    val_scores = []
    for idx, grid_point in enumerate(param_grid):
        grid_point.update(params)
        est.set_params(grid_point)
        scores = cross_validate(est, X, y, groups=None, scoring=scoring, cv=cv_folds, n_jobs=-1, verbose=0, fit_params=None,
                                return_train_score=False, return_estimator=True, error_score='raise-deprecating')

        est_gp_fits.append({'est': scores['estimator'][np.argmax(scores['test_score'])],
                            'val_score': np.max(scores['test_score'])
                           })
    # !!assumption test scores are positive greater the better
    est_gp_fits_top5 = est_gp_fits.sort(reverse=True, key=sort_criterion)[0:5]
    for egpf in est_gp_fits_top5:
        est5.append(egpf['est'])
        val_scores.append(egpf['val_score'])

    return est5, val_scores


def pred(est_type, task, est5, X, y):
    pred_frames = []
    pred_scores = []
    perm_imp = []
    for i, e in enumerate(est5):
        est = e['est']
        ps = e['est'].score(X, y)

        pred_scores.append(ps)
        pred_frames.append(pd.DataFrame(index=y.index.tolist(),
                                        data={'YBOCS_pred': est.predict(X),
                                              'YBOCS_target': y.iloc[:, 0]}))
        if task is 'clf':
            if est_type is 'linear':
                pred_frames[i].insert(1, 'Confidence', est.decision_function(X))
            elif est_type is 'non-linear':
                pred_frames[i].insert(1, 'Confidence', est.predict_proba(X))

        perm_imp.append(perm_imp_test(est=est, base_score=ps, X=X, y=y, n_iter=20))
    return pred_frames, pred_scores, perm_imp


def perm_imp_test(est, base_score, X, y, n_iter=15):
    perm_imp = pd.DataFrame(index=['perm_imp'], columns=[c for c in X.columns.tolist() if c not in gbl.demo_clin_feats])
    for column in perm_imp:
        X_dum = X.copy(deep=True)
        score_diff = 0.0
        for _ in np.arange(n_iter):
            X_dum.loc[:, column] = np.random.permutation(X_dum.loc[:, column])
            score_diff += base_score - est.score(X_dum, y)
        perm_imp.at['perm_imp', column] = score_diff/n_iter
    return perm_imp
