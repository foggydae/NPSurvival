from Survival.Utils import load_val_data
from Survival.Utils import load_score_containers
from Survival.Utils import calc_scores
from Survival.Utils import filename_generator

from Survival.RandomSurvivalForest import RandomSurvivalForest

import numpy as np
import pickle

if __name__ == '__main__':
    
    # get the parameters
    n_trees = [50]
    max_features = [5, 10, 20, 40, 60, 80, 120, 200]
    max_depths = [3, 6, 9, 12]
    pca_flags = [False, True]
    dataset_idxs = [0, 1] # 0: "pancreatitis", 1: "ich", 2: "sepsis"

    train_dfs, test_dfs, dataset_names = load_val_data(dataset_idxs, verbose=True)

    for pca_flag in pca_flags:
        for dataset_idx, dataset_name in enumerate(dataset_names):
            filename = filename_generator("RSF", pca_flag, [dataset_idx])
            concordances, ipecs = load_score_containers([dataset_name], [n_trees, max_features, max_depths])
            print("\nFor the " + dataset_name + " dataset:")

            for dim1, n_tree in enumerate(n_trees):
                for dim2, max_feature in enumerate(max_features):
                    for dim3, max_depth in enumerate(max_depths):
                        print("[LOG] n_tree = {}, max_feature = {}, max_depth = {}".format(
                            n_tree, max_feature, max_depth))

                        tmp_concordances = []
                        tmp_ipecs = []

                        for index, cur_train in enumerate(train_dfs[dataset_name]):
                            model = RandomSurvivalForest(n_trees=n_tree, 
                                max_features=max_feature, max_depth=max_depth, 
                                pca_flag=pca_flag, n_components=int(np.max([20, max_feature*1.2])))
                            model.fit(cur_train, duration_col='LOS', event_col='OUT')
                            concordance, ipec_score = \
                                calc_scores(model, cur_train, test_dfs[dataset_name][index])

                            tmp_concordances.append(concordance)
                            tmp_ipecs.append(ipec_score)

                        avg_concordance = np.average(tmp_concordances)
                        avg_ipec = np.average(tmp_ipecs)

                        print("[LOG] avg. concordance:", avg_concordance)
                        print("[LOG] avg. ipec:", avg_ipec)

                        concordances[dataset_type][dim1][dim2][dim3] = avg_concordance
                        ipecs[dataset_type][dim1][dim2][dim3] = avg_ipec

                        print("-------------------------------------------------------")

                        with open(filename, 'wb') as f:
                            pickle.dump([n_tree, max_features, max_depths, concordances, ipecs], f, pickle.HIGHEST_PROTOCOL)
