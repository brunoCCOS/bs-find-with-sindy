import numpy as np
from scipy.optimize import minimize

def MLSQT(G, b, Lambda,tol = 1e-5):

    w_init= np.linalg.lstsq(G, b, rcond=None)[0]

    # Define the cost function L(λ)
    def cost_function(lambda_val, G, b, w):

        # Compute the active set of indices based on the updated bounds
        norm_diff = lambda j: np.linalg.norm(b) / np.linalg.norm(G[:,j])
        w_lambda = np.array([w[j] if lambda_val * max(1,norm_diff(j)) <= abs(w[j]) <= 1/lambda_val * min(1, norm_diff(j)) else 0 for j in range(len(w))])

        Gw_lambda = np.dot(G, w_lambda)
        Gw_ls =  np.dot(G, w_init)
        norm_diff = np.linalg.norm(Gw_lambda - Gw_ls) / np.linalg.norm(Gw_ls)
        non_zero_indices = np.where(w_lambda)[0]
        cardinality = len(non_zero_indices)
        return norm_diff + cardinality / len(G[0])
    
    optimal_lambda = 0
    cost = np.inf
    for l in Lambda:
        # Find the optimal lambda value
        cost_ = cost_function(l,G, b, w_init)
        if cost_ < cost:
            cost = cost_
            optimal_lambda = l

    print(optimal_lambda)
    w = w_init
    w_ = np.zeros(w.shape)
    while True:
        norm_diff = lambda j: np.linalg.norm(b) / np.linalg.norm(G[:,j])
    
        idx = [j for j in range(len(w)) if optimal_lambda * max(1, norm_diff(j)) <= abs(w[j]) <= 1/optimal_lambda * min(1, norm_diff(j))]

        w_[idx] = np.linalg.lstsq(G[:,idx], b, rcond=None)[0]
        
        if (np.linalg.norm(w_ - w)) < tol:
            w = w_
            break
            
        w = w_
    return w,optimal_lambda


def threshold_remove(data,coef,target,threshold = 0.1,axis=1):
    print(data.shape)
    #Iterate through all terms and force to 0 the ones which does not change the norm of the matrix more than the threshold
    for i in range(len(coef)):
        coef_ = np.delete(coef,i,axis=0)
        data_ = np.delete(data,i,axis=axis)
        matrix_org = data @ coef
        matrix_tg = data_ @ coef_
        if np.abs(np.linalg.norm(matrix_tg) - np.linalg.norm(matrix_org))/np.linalg.norm(matrix_org) < threshold:
            coef[i] = 0. 
    # Optimize the coefficients of the remaining terms
    idx_null = np.where(coef == 0)[0]
    idx_not_null = np.where(coef)
    data_ = np.delete(data,idx_null,axis=axis)
    if len(data_.shape)> 2:
        x = np.reshape(data_,(data_.shape[0]*data_.shape[1],data_.shape[2]))
    else:
        x = data_

    y = np.ravel(target)
    c, r, rank, s = np.linalg.lstsq(x, y, rcond=None)
    coef[idx_not_null] = c
    return coef
