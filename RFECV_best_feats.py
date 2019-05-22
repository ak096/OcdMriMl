import xgboost
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import pandas as pd
from random import randint


def rfecv(t_frame, cv_folds, pat_frame_train, pat_frame_train_y, pat_frame_test, pat_frame_test_y):
    # estimator = xgboost.XGBClassifier(random_state=randint(1, 101))
    # estimator = SVC(C=500, kernel='linear',random_state=randint(1, 101))
    # estimator = GradientBoostingClassifier(random_state=randint(1, 101))
    estimator = AdaBoostClassifier(random_state=randint(1, 101))

    selector = RFECV(estimator, min_features_to_select=10, cv=cv_folds, n_jobs=-1, step=1,
                     verbose=2, scoring='balanced_accuracy')
    selector.fit(pat_frame_train,
                 pat_frame_train_y)
    predictions = selector.predict(pat_frame_test)
    result = pd.DataFrame(index=pat_frame_test_y.index.tolist(),
                          data={'YBOCS_pred': predictions, 'YBOCS_target': pat_frame_test_y.iloc[:, 0]})
    print(result)
    score = selector.score(pat_frame_test, pat_frame_test_y)
    print(score)
    return


feat_xgb_vanilla_min15_clf_ros_resamp = [
'3rd-Ventricle**area',
'CC_Central**area',
'CC_Posterior**area',
'CortexVol**volume',
'Left-Caudate**area',
'Left-Cerebellum-White-Matter**area',
'Left-vessel**area',
'MaskVol-to-eTIV**volume',
'Optic-Chiasm**area',
'Right-Amygdala**area',
'Right-Cerebellum-White-Matter**area',
'Right-Lateral-Ventricle**area',
'SurfaceHoles**volume',
'lh_G&S_cingul-Mid-Ant_area**aparc.a2009s',
'lh_G&S_cingul-Mid-Post_area**aparc.a2009s',
'lh_G&S_cingul-Mid-Post_thickness**aparc.a2009s',
'lh_G&S_cingul-Mid-Post_volume**aparc.a2009s',
'lh_G&S_frontomargin_thickness**aparc.a2009s',
'lh_G&S_frontomargin_volume**aparc.a2009s',
'lh_G_cingul-Post-dorsal_volume**aparc.a2009s',
'lh_G_front_inf-Orbital_volume**aparc.a2009s',
'lh_G_front_inf-Triangul_area**aparc.a2009s',
'lh_G_front_sup_area**aparc.a2009s',
'lh_G_insular_short_area**aparc.a2009s',
'lh_G_orbital_thickness**aparc.a2009s',
'lh_G_pariet_inf-Angular_area**aparc.a2009s',
'lh_G_pariet_inf-Angular_volume**aparc.a2009s',
'lh_G_temp_sup-Lateral_thickness**aparc.a2009s',
'lh_G_temporal_middle_thickness**aparc.a2009s',
'lh_Lat_Fis-ant-Horizont_area**aparc.a2009s',
'lh_Lat_Fis-ant-Horizont_volume**aparc.a2009s',
'lh_Pole_occipital_volume**aparc.a2009s',
'lh_S_central_thickness**aparc.a2009s',
'lh_S_cingul-Marginalis_area**aparc.a2009s',
'lh_S_circular_insula_inf_thickness**aparc.a2009s',
'lh_S_collat_transv_post_volume**aparc.a2009s',
'lh_S_oc_middle&Lunatus_area**aparc.a2009s',
'lh_S_oc_sup&transversal_volume**aparc.a2009s',
'lh_S_orbital_lateral_area**aparc.a2009s',
'lh_S_orbital_lateral_thickness**aparc.a2009s',
'lh_S_postcentral_area**aparc.a2009s',
'lh_S_precentral-sup-part_area**aparc.a2009s',
'lh_S_precentral-sup-part_volume**aparc.a2009s',
'lh_S_temporal_transverse_thickness**aparc.a2009s',
'lh_caudalanteriorcingulate_thickness**aparc',
'lh_inferiortemporal_volume**aparc',
'lh_pericalcarine_volume**aparc',
'lh_precentral_thickness**aparc',
'rh_G&S_cingul-Mid-Post_thickness**aparc.a2009s',
'rh_G&S_frontomargin_thickness**aparc.a2009s',
'rh_G&S_frontomargin_volume**aparc.a2009s',
'rh_G_cingul-Post-ventral_volume**aparc.a2009s',
'rh_G_front_inf-Opercular_thickness**aparc.a2009s',
'rh_G_front_middle_volume**aparc.a2009s',
'rh_G_front_sup_volume**aparc.a2009s',
'rh_G_oc-temp_med-Lingual_thickness**aparc.a2009s',
'rh_G_rectus_volume**aparc.a2009s',
'rh_G_subcallosal_area**aparc.a2009s',
'rh_G_temp_sup-Plan_tempo_area**aparc.a2009s',
'rh_G_temporal_middle_volume**aparc.a2009s',
'rh_Lat_Fis-ant-Vertical_area**aparc.a2009s',
'rh_Lat_Fis-ant-Vertical_volume**aparc.a2009s',
'rh_S_central_thickness**aparc.a2009s',
'rh_S_cingul-Marginalis_area**aparc.a2009s',
'rh_S_circular_insula_sup_thickness**aparc.a2009s',
'rh_S_collat_transv_ant_volume**aparc.a2009s',
'rh_S_front_sup_volume**aparc.a2009s',
'rh_S_orbital-H_Shaped_thickness**aparc.a2009s',
'rh_S_postcentral_area**aparc.a2009s',
'rh_S_postcentral_volume**aparc.a2009s',
'rh_S_precentral-inf-part_area**aparc.a2009s',
'rh_S_precentral-inf-part_volume**aparc.a2009s',
'rh_S_subparietal_thickness**aparc.a2009s',
'rh_cuneus_volume**aparc',
'rh_entorhinal_thickness**aparc',
'rh_entorhinal_volume**aparc',
'rh_frontalpole_volume**aparc',
'rh_fusiform_area**aparc',
'rh_fusiform_thickness**aparc',
'rh_inferiortemporal_volume**aparc',
'rh_insula_area**aparc',
'rh_lingual_area**aparc',
'rh_parahippocampal_volume**aparc',
'rh_parsopercularis_volume**aparc',
'rh_posteriorcingulate_thickness**aparc',
'rh_posteriorcingulate_volume**aparc',
'rh_precuneus_thickness**aparc',
'rh_precuneus_volume**aparc'
]