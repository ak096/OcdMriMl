

def find_best_model(name, estimators_class_per_feats, reg_scoring):
    # if 'reg' in name:
    #     if 'error' in reg_scoring:
    #         descending = False
    #     else:
    #         descending = True
    # elif 'clf' in name:
    #     descending = True
    descending = True
    estimators_class_per_feats.sort(key=lambda x: x[0].best_score_, reverse=descending)
    best = estimators_class_per_feats[0]
    best5 = estimators_class_per_feats[0:4]
    best_model = {'GridObject': best[0], 'est_type': best[1], 'normIdx_train': best[2], 'num_feats': best[3]}

    return best_model, best5
