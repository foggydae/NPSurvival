from npsurvival.FeatureEngineer import FeatureEngineer
from npsurvival.Utils import model_prepare
from npsurvival.Utils import calculate_dataset_size
from npsurvival.Utils import evaluate_predict_result
from npsurvival.IPEC import IPEC

from npsurvival.CoxPHModel import CoxPHModel

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
    lambds = [0.05, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15]

    concordances = {
        "pancreatitis": np.zeros(len(lambds)),
        "ich": np.zeros(len(lambds))
    }
    ipecs = {
        "pancreatitis": np.zeros(len(lambds)),
        "ich": np.zeros(len(lambds))
    }

    for dataset_type in ["pancreatitis", "ich"]:
        cur_trains = train_dfs[dataset_type]
        cur_tests = test_dfs[dataset_type]

        print("\nFor the", dataset_type, "dataset:")
        for row, lambd in enumerate(lambds):
            print("[LOG] lambd = {}".format(lambd))

            tmp_concordances = []
            tmp_ipecs = []
            for index, cur_train in enumerate(cur_trains):
                print(index, end=" ")
                cur_test = cur_tests[index]
                ipec = IPEC(cur_train, g_type="All_One", t_thd=0.8, 
                    t_step="obs", time_col='LOS', death_identifier='OUT')

                model = CoxPHModel(alpha=1, lambda_=lambd)
                model.fit(cur_train, duration_col='LOS', event_col='OUT')
                test_time_median_pred = model.pred_median_time(cur_test)
                proba_matrix = \
                    model.pred_proba(cur_test, time=ipec.get_check_points())

                concordance = evaluate_predict_result(test_time_median_pred, 
                    cur_test, print_result=False)
                ipec_score = ipec.calc_ipec(proba_matrix, 
                    list(cur_test["LOS"]), list(cur_test["OUT"]))

                tmp_concordances.append(concordance)
                tmp_ipecs.append(ipec_score)

            avg_concordance = np.average(tmp_concordances)
            avg_ipec = np.average(tmp_ipecs)

            print("[LOG] avg. concordance:", avg_concordance)
            print("[LOG] avg. ipec:", avg_ipec)

            concordances[dataset_type][row] = avg_concordance
            ipecs[dataset_type][row] = avg_ipec

            print("-------------------------------------------------------")

    with open('COXModel/COX.pickle', 'wb') as f:
        pickle.dump([concordances, ipecs], f, pickle.HIGHEST_PROTOCOL)
