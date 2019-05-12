

def find_best_model(name, feats_estClass_models, reg_scoring):
    # if 'reg' in name:
    #     if 'error' in reg_scoring:
    #         descending = False
    #     else:
    #         descending = True
    # elif 'clf' in name:
    #     descending = True
    descending = True
    feats_estClass_models.sort(key=lambda x: x[0].best_score_, reverse=descending)
    best = feats_estClass_models[0]
    best5 = feats_estClass_models[0:4]
    best_model = {'EstObject': best[0], 'est_type': best[1], 'normIdx_train': best[2], 'num_feats': best[3]}

    return best_model, best5
