

def find_best_model(name, estimators_class_per_feats):
    if 'reg' in name:
        r = False
    elif 'clr' in name:
        r = True
    estimators_class_per_feats.sort(key=lambda x: x[0].best_score_, reverse=r)
    best = estimators_class_per_feats[0]
    best_model = {'GridObject': best[0], 'est_type': best[1], 'normIdx_train': best[2], 'num_feats': best[3]}

    return best_model
