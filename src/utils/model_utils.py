import itertools

import numpy as np
import pysindy as ps
from utils.print_utils import print_model
from sklearn.model_selection import KFold


def threshold_remove(data,coef,target,threshold = 0.1,axis=1):
    #Iterate through all terms and force to 0 the ones which does not change the norm of the matrix more than the threshold
    for i in range(len(coef)):
        coef_ = np.delete(coef,i,axis=0)
        data_ = np.delete(data,i,axis=axis)
        matrix_org = data @ coef
        matrix_tg = data_ @ coef_
        if np.abs(np.linalg.norm(matrix_tg) - np.linalg.norm(matrix_org))/np.linalg.norm(matrix_org) < threshold:
            coef[i] = 0. 
    # Optimize the coefficients of the remaining terms
    idx_not_null = np.where(coef)
    data_ = data_[:,:,idx_not_null[0]]
    x = np.reshape(data_,(data_.shape[0]*data_.shape[1],data_.shape[2]))
    y = np.ravel(target)
    c, r, rank, s = np.linalg.lstsq(x, y, rcond=None)
    coef[idx_not_null] = c
    return coef

def grind_hyper_search(u, u_dot, lib, opt, param_grid, num_folds=3, **model_keyargs) -> None:
    # Create an empty dictionary to store results
    results = {}
    coefs = {}
    
    # Set the random seed for reproducibility (optional)
    np.random.seed(42)

    # Define the k-fold cross-validation object
    kf = KFold(n_splits=num_folds)

    keys, values = zip(*param_grid.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)] # Create a list of dicts with all combinations

    # Loop over different alpha values and threshold values
    for comb in permutations_dicts:
        scores_n = []
        scores = []  # To store scores for each fold

        for train_index, test_index in kf.split(u):
            train_lib = u[train_index]
            test_lib = u[test_index]
            if u_dot is not None:
                train_set = u_dot[train_index]
                test_set = u_dot[test_index]
            else:
                train_set = None
                test_set = None
            # Create an instance of the optimizer with the params of grind
            optimizer = opt(**comb)

            # Create an instance of the SINDy model with the optimizer
            model = ps.SINDy(feature_library=lib, optimizer=optimizer)
            model.fit(train_lib, x_dot=train_set, **model_keyargs)

            # Calculate the score using u_dot 
            score_n = model.score(test_lib, x_dot=test_set)
            pred = model.predict(test_lib)
            score = (np.sqrt(np.sum((pred.flatten() - test_set.flatten())**2).mean()))
            scores_n.append(score_n)
            scores.append(score)

        # Store the average score across folds
        coefs[tuple(comb.values())] = model.coefficients()[0]
        results[tuple(comb.values())] = np.mean(scores)

    # Find the hyperparameters with the best performance
    best_hyperparameters = min(results, key=results.get)
    best_score = results[best_hyperparameters]

    print(f"Best Hyperparameters:{[f'{keys[i]} = {best_hyperparameters[i]}' for i in range(len(best_hyperparameters))]}")
    print("Best Score:", best_score)
    print_model(coefs[best_hyperparameters], lib.get_feature_names())