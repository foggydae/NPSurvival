from Survival.Utils import load_val_data
from Survival.Utils import load_score_containers
from Survival.Utils import calc_scores
from Survival.Utils import filename_generator

from Survival.KNNKaplanMeier import KNNKaplanMeier

import numpy as np
import pickle

if __name__ == '__main__':
    
    # get the parameters
    n_neighbors = [3, 5, 10, 15, 20, 30, 50, 80, 120, 200]
    pca_flag = False
    dataset_idxs = [2] # 0: "pancreatitis", 1: "ich", 2: "sepsis"
    filename = filename_generator("KNN", pca_flag, dataset_idxs)


    train_dfs, test_dfs, dataset_names = load_val_data(dataset_idxs)
    concordances, ipecs = load_score_containers(dataset_names, [n_neighbors])

    for dataset_name in dataset_names:
        print("\nFor the " + dataset_name + " dataset:")

        for row, n_neighbor in enumerate(n_neighbors):
            print("[LOG] n_neighbor = {}".format(n_neighbor))

            tmp_concordances = []
            tmp_ipecs = []

            for index, cur_train in enumerate(train_dfs[dataset_name]):
                model = KNNKaplanMeier(n_neighbors=n_neighbor, 
                    pca_flag=pca_flag, n_components=20)
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
                pickle.dump([n_neighbors, concordances, ipecs], f, pickle.HIGHEST_PROTOCOL)
