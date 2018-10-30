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
    dataset_idxs = [0, 1, 2] # 0: "pancreatitis", 1: "ich", 2: "sepsis"
    filename = filename_generator("WBR", pca_flag, dataset_idxs)
    
    train_dfs, test_dfs, dataset_names = load_val_data(dataset_idxs, verbose=True)
    concordances, ipecs = load_score_containers(dataset_names, [n_components])

