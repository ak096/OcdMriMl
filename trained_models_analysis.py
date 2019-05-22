from prediction_reporting import predict_report
import gbl


def find_best_models(name, feats_estClass_models, reg_scoring):
    # if 'reg' in name:
    #     if 'error' in reg_scoring:
    #         descending = False
    #     else:
    #         descending = True
    # elif 'clf' in name:
    #     descending = True
    descending = True
    feats_estClass_models.sort(key=lambda x: x[0].best_score_, reverse=descending)

    best_models_list = [{'EstObject': elt[0], 'est_type': elt[1], 'normIdx_train': elt[2], 'num_feats': elt[3],
                         't_feats_idx': list(elt[4])} for elt in feats_estClass_models[0:4]]

    return best_models_list


def models_to_results(models_all, pat_frame_test_reg_norms, pat_frame_test_clf_norms,
                      pat_frame_test_y_reg, pat_frame_test_y_clf, reg_scoring):

    for key, value in models_all.items():
        if value:
            if 'reg' in key:
                ec = 'reg'
                pat_frame_test_norms = pat_frame_test_reg_norms
                pat_frame_test_y = pat_frame_test_y_reg
            elif 'clf' in key:
                ec = 'clf'
                pat_frame_test_norms = pat_frame_test_clf_norms
                pat_frame_test_y = pat_frame_test_y_clf

            bm5 = find_best_models(key, value, reg_scoring)
            bm = bm5[0]

            pat_frame_test_norm = pat_frame_test_norms[bm['normIdx_train']]

            if key == gbl.t_r:
                gbl.feat_sets_best_train[key] = gbl.t_frame_global.columns[bm['t_feats_idx']].tolist() \
                                                + gbl.demo_clin_feats
            elif key == gbl.t_c:
                gbl.feat_sets_best_train[key] = gbl.t_frame_global.columns[bm['t_feats_idx']].tolist() \
                                                + gbl.demo_clin_feats

            ft = gbl.feat_sets_best_train[key]
            print(ft)
            pr = predict_report(key, bm, pat_frame_test_norm, ft, pat_frame_test_y, ec)

            gbl.best_models_results[key] = {'features': ft, 'est_class': ec, 'best_model': bm, 'pred_results': pr, 'bm5': bm5}

    return
