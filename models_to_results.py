from prediction_reporting import predict_report
from find_best_model import find_best_model
import gbl


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

            bm, bm5 = find_best_model(key, value, reg_scoring)

            pat_frame_test_norm = pat_frame_test_norms[bm['normIdx_train']]

            if key == gbl.t_r:
                gbl.feat_sets_best_train[key] = gbl.t_frame_perNorm_list[bm['normIdx_train']].columns[0:bm['num_feats'] - 4].tolist() \
                                                + gbl.demo_clin_feats
            elif key == gbl.t_c:
                gbl.feat_sets_best_train[key] = gbl.t_frame_perNorm_list[bm['normIdx_train']].columns[0:bm['num_feats'] - 4].tolist() \
                                                + gbl.demo_clin_feats

            ft = gbl.feat_sets_best_train[key]
            print(ft)
            pr = predict_report(key, bm, pat_frame_test_norm, ft, pat_frame_test_y, ec)

            gbl.best_models_results[key] = {'features': ft, 'est_class': ec, 'best_model': bm, 'pred_results': pr, 'bm5': bm5}

    return
