from Survival.Utils import load_val_data
from Survival.Utils import calc_scores
from Survival.Utils import filename_generator

from Survival.NeuralNetworkCox import NeuralNetworkCox

import numpy as np
import pickle

if __name__ == '__main__':

    dataset_idxs = [0]  # 0: "pancreatitis", 1: "ich", 2: "sepsis"
    train_dfs, test_dfs, unique_times, dataset_names = \
        load_val_data(dataset_idxs, True)
    first_layers = [8, 16, 32, 64, 96, 128, 192, 256]
    lmbdas = [0., 0.01, 0.02]
    dataset_name = dataset_names[0]
    dataset_idx = dataset_idxs[0]
    filename = filename_generator("NNC", False, [dataset_idx])
    concordances = {}
    ipecs = {}

    print("\nFor the " + dataset_name + " dataset:")

    for row, first_layer in enumerate(first_layers):
        for col, lmbda in enumerate(lmbdas):
            print("[LOG] first_layer = {}, lmbda = {}".format(
                first_layer, lmbda))

            tmp_concordances = []
            tmp_ipecs = []

            for index, cur_train in enumerate(train_dfs[dataset_name]):
                print(index, end=" ")
                cur_test = test_dfs[dataset_name][index]
                model = NeuralNetworkCox(first_layer_size=first_layer,
                                         lmbda=lmbda, verbose=0)
                model.fit(cur_train, duration_col='LOS', event_col='OUT')
                concordance, ipec_score = \
                    calc_scores(model, cur_test,unique_times[dataset_name])
                print(concordance, ipec_score[int(len(ipec_score) * 0.8)])

                tmp_concordances.append(concordance)
                tmp_ipecs.append(ipec_score)

            avg_concordance = np.average(tmp_concordances)
            avg_ipec = np.average(tmp_ipecs, axis=0)
            print("[LOG] avg. concordance:", avg_concordance)
            print("[LOG] avg. ipec:", avg_ipec[int(len(avg_ipec) * 0.8)])

            concordances[(first_layer, lmbda)] = avg_concordance
            ipecs[(first_layer, lmbda)] = avg_ipec

            print("-------------------------------------------------------")

            with open(filename, 'wb') as f:
                pickle.dump([first_layers, lmbdas, concordances, ipecs], f,
                            pickle.HIGHEST_PROTOCOL)
