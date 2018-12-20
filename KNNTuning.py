from Survival.Utils import load_val_data
from Survival.Utils import calc_scores
from Survival.Utils import filename_generator

from Survival.KNNKaplanMeier import KNNKaplanMeier

import numpy as np
import pickle

if __name__ == '__main__':
    
    # get the parameters
    n_neighbors = [3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140,
                   160, 180, 200]
    pca_flags = [True, False]
    dataset_idxs = [0, 1, 2] # 0: "pancreatitis", 1: "ich", 2: "sepsis"

    train_dfs, test_dfs, unique_times, dataset_names = \
        load_val_data(dataset_idxs, verbose=False)

    for pca_flag in pca_flags:
        for dataset_idx, dataset_name in enumerate(dataset_names):
            filename = filename_generator("KNN", pca_flag, [dataset_idx])
            concordances = {}
            ipecs = {}
            print("\nFor the " + dataset_name + " dataset:")

            for row, n_neighbor in enumerate(n_neighbors):
                print("[LOG] n_neighbor = {}".format(n_neighbor))

                tmp_concordances = []
                tmp_ipecs = []

                for index, cur_train in enumerate(train_dfs[dataset_name]):
                    cur_test = test_dfs[dataset_name][index]
                    model = KNNKaplanMeier(n_neighbors=n_neighbor,
                        pca_flag=pca_flag, n_components=20)
                    model.fit(cur_train, duration_col='LOS', event_col='OUT')
                    concordance, ipec_score = \
                        calc_scores(model, cur_test, unique_times[dataset_name])
                    print(concordance, ipec_score[int(len(ipec_score) * 0.8)])

                    tmp_concordances.append(concordance)
                    tmp_ipecs.append(ipec_score)

                avg_concordance = np.average(tmp_concordances)
                avg_ipec = np.average(tmp_ipecs, axis=0)
                print("[LOG] avg. concordance:", avg_concordance)
                print("[LOG] avg. ipec:", avg_ipec[int(len(avg_ipec) * 0.8)])

                concordances[n_neighbor] = avg_concordance
                ipecs[n_neighbor] = avg_ipec

                print("-------------------------------------------------------")

                with open(filename, 'wb') as f:
                    pickle.dump([n_neighbors, concordances, ipecs], f,
                                pickle.HIGHEST_PROTOCOL)
