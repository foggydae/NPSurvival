from Survival.Utils import load_val_data
from Survival.Utils import load_score_containers
from Survival.Utils import calc_scores
from Survival.Utils import filename_generator

from Survival.AalenAdditiveModel import AalenAdditiveModel

import numpy as np
import pickle

if __name__ == '__main__':
    
    ## set the parameters
    coef_penalizers = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25, 0.27, 0.29, 0.32, 0.35, 0.38, 0.4, 0.42, 0.45, 0.48, 0.5, 0.53, 0.56, 0.6, 0.63, 0.66, 0.7, 0.73, 0.76, 0.8, 0.83, 0.86, 0.9]
    pca_flags = [False, True]
    dataset_idxs = [0, 1] # 0: "pancreatitis", 1: "ich", 2: "sepsis"
    
    train_dfs, test_dfs, dataset_names = load_val_data(dataset_idxs, verbose=False)

    for pca_flag in pca_flags:
        for dataset_idx, dataset_name in enumerate(dataset_names):
            filename = filename_generator("AAM", pca_flag, [dataset_idx])
            concordances, ipecs = load_score_containers([dataset_name], [coef_penalizers])
            print("\nFor the " + dataset_name + " dataset:")

            for row, coef_penalizer in enumerate(coef_penalizers):
                print("[LOG] coef_penalizer = {}".format(coef_penalizer))

                tmp_concordances = []
                tmp_ipecs = []

                for index, cur_train in enumerate(train_dfs[dataset_name]):
                    model = AalenAdditiveModel(coef_penalizer=coef_penalizer, 
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
                    pickle.dump([coef_penalizers, concordances, ipecs], f, pickle.HIGHEST_PROTOCOL)
