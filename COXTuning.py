from Survival.Utils import load_val_data
from Survival.Utils import calc_scores
from Survival.Utils import filename_generator

from Survival.CoxPHModel import CoxPHModel

import numpy as np
import pickle

if __name__ == '__main__':
    
    # get the parameters
    lambds = [0.0001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
              0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
    pca_flags = [True, False]
    dataset_idxs = [0] # 0: "pancreatitis", 1: "ich", 2: "sepsis"

    train_dfs, test_dfs, unique_times, dataset_names = \
        load_val_data(dataset_idxs, verbose=False)

    for pca_flag in pca_flags:
        for dataset_idx, dataset_name in enumerate(dataset_names):
            filename = filename_generator("COX", pca_flag, [dataset_idx])
            concordances = {}
            ipecs = {}
            print("\nFor the " + dataset_name + " dataset:")

            for row, lambd in enumerate(lambds):
                print("[LOG] lambda = {}".format(lambd))

                tmp_concordances = []
                tmp_ipecs = []

                for index, cur_train in enumerate(train_dfs[dataset_name]):
                    cur_test = test_dfs[dataset_name][index]
                    model = CoxPHModel(lambda_=lambd, alpha=1,
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

                concordances[lambd] = avg_concordance
                ipecs[lambd] = avg_ipec

                print("-------------------------------------------------------")

                with open(filename, 'wb') as f:
                    pickle.dump([lambds, concordances, ipecs], f,
                                pickle.HIGHEST_PROTOCOL)
