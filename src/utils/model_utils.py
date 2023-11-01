import itertools

import numpy as np
import pysindy as ps
from utils.print_utils import print_model
from sklearn.model_selection import train_test_split, KFold

import numpy.linalg as la

def best_hyperparameters(X, u, lambdas, thresholds, n_splits=5):
    """
    Find the best lambda and threshold for SINDy using cross-validation.

    Parameters:
    - X: The independent variable data
    - u: The dependent variable data
    - lambdas: List of lambda values to test
    - thresholds: List of threshold values to test
    - n_splits: Number of splits for cross-validation

    Returns:
    - best_lambda: The best lambda value
    - best_threshold: The best threshold value
    """

    # Split data into training and validation sets
    X_train, X_val, u_train, u_val = train_test_split(X, u, test_size=0.2, random_state=42)
    print(X_train.shape,u_train.shape)
    best_score = float('inf')
    best_lambda = None
    best_threshold = None

    for l in lambdas:
        for t in thresholds:
            # Define the SINDy optimizer with the current lambda and threshold
            optimizer = ps.optimizers.STLSQ(threshold=t, alpha=l).fit(X_train,u_train)
            
            
            # Predict on the validation set
            u_pred = optimizer.coef_[0] @ X_val
            
            # Calculate the mean squared error on the validation set
            mse = np.mean((u_val - u_pred)**2)
            
            # Update the best hyperparameters if current MSE is lower
            if mse < best_score:
                best_score = mse
                best_lambda = l
                best_threshold = t

    return best_lambda, best_threshold


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


def iter_psdn(u,
            lib,
            sigma_estimate,
            A,
            max_iter=10,
            alpha=.1,
            center_Theta=False,
            check_diverge=False):
    """Perform projection-based denoising.

    Args:
        u (d X N np.array): description
        lib: Library object to construct the Theta matrix
        sigma_estimate (d np.array):
        A ():
        max_iter (int):
        alpha (float in [0,1]):

    Returns:
        type: description


    """
    N = np.size(u, 1)
    m = np.size(u, 0)
    u_proj = np.copy(u)
    u_err_vec = []
    sigma_vec = []
    sum_vec = []

    for i in range(max_iter):

        Theta_temp = lib.fit_transform(u_proj)
        if center_Theta:
            Theta_temp = mlu.center_Theta(Theta_temp, d, m,
                                            sigma_estimate[0]**2)

        Phi = A @ Theta_temp

        Phi = np.hstack((np.ones(N).reshape(-1, 1), Phi))

        # Use SVD to perform projeciton
        U = la.svd(Phi, full_matrices=False)[0]
        P_Phi = U @ U.T
        u_proj_new = alpha * (P_Phi @ u_proj.T).T + (1 - alpha) * u_proj

        # Record mean of error
        sum_vec.append(1 / np.sqrt(N) * np.sum(u_proj_new - u, axis=1))

        # Record variance history
        sigma_pred = 1 / np.sqrt(N) * la.norm(u_proj_new - u, axis=1)
        sigma_vec.append(sigma_pred)

        # Check for divergence and break if sigma_pred is too large
        if check_diverge:
            update = (sigma_pred < sigma_estimate)
            if sum(update) == 0:
                print('WARNING: HIT MAX SIGMA')
                break
            # If varaince too large don't perform projection
            u_proj_new[update == 0, :] = u_proj[update == 0, :]

        n = np.min((np.size(u_proj_new, 1), np.size(u_proj, 1)))
        true_norm = la.norm(u_proj_new[:, :n], axis=1)
        conv_norm = la.norm(u_proj_new[:, :n] - u_proj[:, :n], axis=1) / true_norm

        if np.max(conv_norm) < 1e-8:

            print('Converged.')
            break

        u_proj = np.copy(u_proj_new)

    return u_proj, la.cond(Phi)

