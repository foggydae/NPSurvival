from Survival.Utils import load_val_data
from Survival.Utils import load_score_containers
from Survival.Utils import calc_scores
from Survival.Utils import filename_generator

from Survival.WeibullRegressionModel import WeibullRegressionModel

import numpy as np
import pickle

if __name__ == '__main__':
    
    ## set the parameters
    n_components = [10, 15, 17, 19, 25, 30]
    pca_flag = True
    dataset_idxs = [0] # 0: "pancreatitis", 1: "ich", 2: "sepsis"
    filename = filename_generator("WBR", pca_flag, dataset_idxs)

    
    train_dfs, test_dfs, dataset_names = load_val_data(dataset_idxs)
    concordances, ipecs = load_score_containers(dataset_names, [n_components])

    for dataset_name in dataset_names:
        print("\nFor the " + dataset_name + " dataset:")

        for row, n_component in enumerate(n_components):
            print("[LOG] n_component = {}".format(n_component))

            tmp_concordances = []
            tmp_ipecs = []

            for index, cur_train in enumerate(train_dfs[dataset_name]):
                model = WeibullRegressionModel(pca_flag=pca_flag, n_components=n_component)
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

            with open(filename, 'wb') as f:
                pickle.dump([n_components, concordances, ipecs], f, pickle.HIGHEST_PROTOCOL)
