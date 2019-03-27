from operator import itemgetter


def find_best_model(estimators_class_per_feats):
    estimators_class_per_feats.sort(key=lambda x: x[0].best_score_)
    best = estimators_class_per_feats[0]
    best_model = {'GridObject': best[0], 'est_type': best[1], 'normType_train': best[2], 'num_feats': best[3]}

    return best_model
