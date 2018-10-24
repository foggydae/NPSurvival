from Survival.FeatureEngineer import FeatureEngineer
from Survival.Utils import model_prepare
from Survival.Utils import calculate_dataset_size
from Survival.Utils import evaluate_predict_result
from Survival.IPEC import IPEC

from Survival.CoxPHModel import CoxPHModel
from Survival.KNNKaplanMeier import KNNKaplanMeier
from Survival.AalenAdditiveModel import AalenAdditiveModel
from Survival.RandomSurvivalForest import RandomSurvivalForest

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
    n_trees = [20, 50, 100]
    max_features = [10, 20]
    max_depths = [3, 6, 10]

    concordances = {
        "pancreatitis": {20: np.zeros((len(max_features), len(max_depths))), 
                         50: np.zeros((len(max_features), len(max_depths))),
                         100: np.zeros((len(max_features), len(max_depths)))},
        "ich": {20: np.zeros((len(max_features), len(max_depths))), 
                50: np.zeros((len(max_features), len(max_depths))),
                100: np.zeros((len(max_features), len(max_depths)))}
    }

    ipecs = {
        "pancreatitis": {20: np.zeros((len(max_features), len(max_depths))), 
                         50: np.zeros((len(max_features), len(max_depths))),
                         100: np.zeros((len(max_features), len(max_depths)))},
        "ich": {20: np.zeros((len(max_features), len(max_depths))), 
                50: np.zeros((len(max_features), len(max_depths))),
                100: np.zeros((len(max_features), len(max_depths)))}
    }

    for dataset_type in ["pancreatitis", "ich"]:
        print("For the", dataset_type, "dataset:\n")
        for n_tree in n_trees:
            for row, max_feature in enumerate(max_features):
                for col, max_depth in enumerate(max_depths):
                    print("[LOG] n_tree = {}, max_feature = {}, max_depth = {}".format(
                        n_tree, max_feature, max_depth))

                    tmp_concordances = []
                    tmp_ipecs = []

                    for index in range(len(train_dfs[dataset_type])):
                        model = RandomSurvivalForest(n_trees=n_tree, 
                            max_features=max_feature, max_depth=max_depth)
                        model.fit(train_dfs[dataset_type][index], duration_col='LOS', event_col='OUT')
                        test_time_median_pred = model.pred_median_time(test_dfs[dataset_type][index])
                        concordance = evaluate_predict_result(test_time_median_pred, 
                            test_dfs[dataset_type][index], print_result=False)
                        tmp_concordances.append(concordance)
                        ipec = IPEC(train_dfs[dataset_type][index], model.pred_proba, 
                            g_type="All_One", t_thd=0.8, t_step="obs")
                        ipec_score = ipec.avg_ipec(test_dfs[dataset_type][index], num_workers=2, 
                            print_result=False)
                        tmp_ipecs.append(ipec_score)

                    avg_concordance = np.average(tmp_concordances)
                    avg_ipec = np.average(tmp_ipecs)

                    print("[LOG] avg. concordance:", avg_concordance)
                    print("[LOG] avg. ipec:", avg_ipec)

                    concordances[dataset_type][n_tree][row][col] = avg_concordance
                    ipecs[dataset_type][n_tree][row][col] = avg_ipec

                    print("-------------------------------------------------------")

                with open('RSF.pickle', 'wb') as f:
                    pickle.dump([concordances, ipecs], f, pickle.HIGHEST_PROTOCOL)
