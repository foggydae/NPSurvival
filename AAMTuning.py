from Survival.Utils import load_val_data
from Survival.Utils import load_score_containers
from Survival.Utils import calc_scores

from Survival.AalenAdditiveModel import AalenAdditiveModel

import numpy as np
import pickle

if __name__ == '__main__':
    
    ## set the parameters
    coef_penalizers = [0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6]

    # 0: "pancreatitis", 1: "ich", 2: "sepsis"
    train_dfs, test_dfs, dataset_names = load_val_data([2])
    concordances, ipecs = load_score_containers(dataset_names, [coef_penalizers])

    for dataset_name in dataset_names:
        print("\nFor the " + dataset_name + " dataset:")

        for row, coef_penalizer in enumerate(coef_penalizers):
            print("[LOG] coef_penalizer = {}".format(coef_penalizer))

            tmp_concordances = []
            tmp_ipecs = []

            for index, cur_train in enumerate(train_dfs[dataset_name]):
                model = AalenAdditiveModel(coef_penalizer=coef_penalizer, pca_flag=True)
                model.fit(cur_train, duration_col='LOS', event_col='OUT')
                concordance, ipec_score = \
                    calc_scores(model, cur_train, test_dfs[dataset_name][index])

                tmp_concordances.append(concordance)
                tmp_ipecs.append(ipec_score)

            avg_concordance = np.average(tmp_concordances)
            avg_ipec = np.average(tmp_ipecs)
            print("[LOG] avg. concordance:", avg_concordance)
            print("[LOG] avg. ipec:", avg_ipec)

            concordances[dataset_name][row] = avg_concordance
            ipecs[dataset_name][row] = avg_ipec

            print("-------------------------------------------------------")

    with open('AAM_results/AAM_w_pca.pickle', 'wb') as f:
        pickle.dump([coef_penalizers, concordances, ipecs], f, pickle.HIGHEST_PROTOCOL)
