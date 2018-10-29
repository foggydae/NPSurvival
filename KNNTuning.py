from Survival.FeatureEngineer import FeatureEngineer
from Survival.Utils import model_prepare
from Survival.Utils import calculate_dataset_size
from Survival.Utils import evaluate_predict_result
from Survival.IPEC import IPEC

from Survival.KNNKaplanMeier import KNNKaplanMeier

import numpy as np
import pickle

if __name__ == '__main__':
    
    fe = FeatureEngineer(verbose=False)
    sources = fe.get_diseases_list()

    # load data
    train_dfs = {"pancreatitis": [], "ich": []}
    test_dfs = {"pancreatitis": [], "ich": []}
    ## pancreatitis datasets
    for i in range(50):
        patient_dict, feature_set, train_id_list, test_id_list = \
            fe.load_data_as_dict(src_idx=0, 
                file_prefix="cross_val/10-5fold_"+str(i)+"_", 
                low_freq_event_thd=0.03, 
                low_freq_value_thd=0.01)
        train_df, test_df, feature_list = \
            model_prepare(patient_dict, feature_set, train_id_list, test_id_list)
        train_dfs["pancreatitis"].append(train_df)
        test_dfs["pancreatitis"].append(test_df)

    ## ich datasets
    for i in range(5):
        patient_dict, feature_set, train_id_list, test_id_list = \
            fe.load_data_as_dict(src_idx=1, 
                file_prefix="cross_val/1-5fold_"+str(i)+"_", 
                low_freq_event_thd=0.03, 
                low_freq_value_thd=0.01)
        train_df, test_df, feature_list = \
            model_prepare(patient_dict, feature_set, train_id_list, test_id_list)
        train_dfs["ich"].append(train_df)
        test_dfs["ich"].append(test_df)

    # get the parameters
    n_neighbors = [3, 5, 10, 15, 20, 30, 50, 80, 120, 200]

    concordances_wo_pca = {
        "pancreatitis": np.zeros(len(n_neighbors)),
        "ich": np.zeros(len(n_neighbors))
    }
    ipecs_wo_pca = {
        "pancreatitis": np.zeros(len(n_neighbors)),
        "ich": np.zeros(len(n_neighbors))
    }
    concordances_w_pca = {
        "pancreatitis": np.zeros(len(n_neighbors)),
        "ich": np.zeros(len(n_neighbors))
    }
    ipecs_w_pca = {
        "pancreatitis": np.zeros(len(n_neighbors)),
        "ich": np.zeros(len(n_neighbors))
    }

    for dataset_type in ["pancreatitis", "ich"]:
        cur_trains = train_dfs[dataset_type]
        cur_tests = test_dfs[dataset_type]
        print("\nFor the", dataset_type, "dataset:")

        for row, n_neighbor in enumerate(n_neighbors):
            print("[LOG] n_neighbor = {}".format(n_neighbor))

            tmp_concordances_wo_pca = []
            tmp_ipecs_wo_pca = []
            tmp_concordances_w_pca = []
            tmp_ipecs_w_pca = []
            for index, cur_train in enumerate(cur_trains):
                print(index, end=" ")
                cur_test = cur_tests[index]
                ipec = IPEC(cur_train, g_type="All_One", t_thd=0.8, 
                    t_step="obs", time_col='LOS', death_identifier='OUT')

                # without PCA
                model = KNNKaplanMeier(n_neighbors=n_neighbor)
                model.fit(cur_train, duration_col='LOS', event_col='OUT')
                test_time_median_pred = model.pred_median_time(cur_test)
                proba_matrix = \
                    model.pred_proba(cur_test, time=ipec.get_check_points())

                concordance = evaluate_predict_result(test_time_median_pred, 
                    cur_test, print_result=False)
                ipec_score = ipec.calc_ipec(proba_matrix, 
                    list(cur_test["LOS"]), list(cur_test["OUT"]))

                tmp_concordances_wo_pca.append(concordance)
                tmp_ipecs_wo_pca.append(ipec_score)

                # with PCA
                model = KNNKaplanMeier(n_neighbors=n_neighbor, 
                    pca_flag=True, n_components=20)
                model.fit(cur_train, duration_col='LOS', event_col='OUT')
                test_time_median_pred = model.pred_median_time(cur_test)
                proba_matrix = \
                    model.pred_proba(cur_test, time=ipec.get_check_points())

                concordance = evaluate_predict_result(test_time_median_pred, 
                    cur_test, print_result=False)
                ipec_score = ipec.calc_ipec(proba_matrix, 
                    list(cur_test["LOS"]), list(cur_test["OUT"]))

                tmp_concordances_w_pca.append(concordance)
                tmp_ipecs_w_pca.append(ipec_score)


            avg_concordance_wo_pca = np.average(tmp_concordances_wo_pca)
            avg_ipec_wo_pca = np.average(tmp_ipecs_wo_pca)
            print("[LOG] avg. concordance w/o pca:", avg_concordance_wo_pca)
            print("[LOG] avg. ipec w/o pca:", avg_ipec_wo_pca)
            concordances_wo_pca[dataset_type][row] = avg_concordance_wo_pca
            ipecs_wo_pca[dataset_type][row] = avg_ipec_wo_pca

            avg_concordance_w_pca = np.average(tmp_concordances_w_pca)
            avg_ipec_w_pca = np.average(tmp_ipecs_w_pca)
            print("[LOG] avg. concordance w/ pca:", avg_concordance_w_pca)
            print("[LOG] avg. ipec w/ pca:", avg_ipec_w_pca)
            concordances_w_pca[dataset_type][row] = avg_concordance_w_pca
            ipecs_w_pca[dataset_type][row] = avg_ipec_w_pca

            print("-------------------------------------------------------")

    with open('KNN_results/KNN_wo_pca.pickle', 'wb') as f:
        pickle.dump([n_neighbors, concordances_wo_pca, ipecs_wo_pca], f, pickle.HIGHEST_PROTOCOL)

    with open('KNN_results/KNN_w_pca.pickle', 'wb') as f:
        pickle.dump([n_neighbors, concordances_w_pca, ipecs_w_pca], f, pickle.HIGHEST_PROTOCOL)

