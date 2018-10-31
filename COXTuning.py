from Survival.Utils import load_val_data
from Survival.Utils import load_score_containers
from Survival.Utils import calc_scores
from Survival.Utils import filename_generator

from Survival.CoxPHModel import CoxPHModel

import numpy as np
import pickle

if __name__ == '__main__':
    
    # get the parameters
    lambds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15]
    pca_flag = False
    dataset_idxs = [2] # 0: "pancreatitis", 1: "ich", 2: "sepsis"
    filename = filename_generator("COX", pca_flag, dataset_idxs)


    train_dfs, test_dfs, dataset_names = load_val_data(dataset_idxs, verbose=True)
    concordances, ipecs = load_score_containers(dataset_names, [lambds])

    for dataset_name in dataset_names:
        print("\nFor the " + dataset_name + " dataset:")

        for row, lambd in enumerate(lambds):
            print("[LOG] lambda = {}".format(lambd))

            tmp_concordances = []
            tmp_ipecs = []

            for index, cur_train in enumerate(train_dfs[dataset_name]):
                model = CoxPHModel(alpha=1, lambda_=lambd, 
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
                pickle.dump([lambds, concordances, ipecs], f, pickle.HIGHEST_PROTOCOL)
