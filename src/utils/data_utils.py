import pandas as pd
import numpy as np
from scipy.stats import norm

def load_data(filename):
    df = pd.read_csv(filename)
    return df

def subsample_f(u,t,x,timesteps):
    num_points_to_sample = timesteps

    # Generate random indices
    random_rows = np.random.choice(u.shape[0], num_points_to_sample, replace=True)
    # random_columns = np.concatenate(([0,-1],np.random.choice(u.shape[1], num_points_to_sample-2, replace=False)))

    # Subsample the matrix
    subsampled_u = np.ravel(u[random_rows,np.linspace(0,timesteps-1,timesteps,dtype=int)])
    subsampled_x = x[random_rows]
    
    return subsampled_u,subsampled_x,random_rows

def build_cross_library(Theta,Theta_names):
    m = Theta.shape[1]
    for i in range(m):
        for j in range(i+1,m):
            Theta_names = np.hstack((Theta_names,Theta_names[i] + '*' + Theta_names[j]))
            Theta = np.hstack((Theta,(Theta[:,i]*Theta[:,j]).reshape(-1,1)))
    return Theta, Theta_names

def build_final_condition(X,Y,K):
    x_min,x_max = np.min(X[:,0]),np.max(X[:,0])
    final_condition_points = np.vstack([np.linspace(x_min,x_max,100),np.ones(100)]).T
    final_condition_values = np.array([np.max([0,x-K]) for x in final_condition_points[:,0]])

    new_X = np.vstack((X,final_condition_points))
    new_Y = np.hstack((Y,final_condition_values))
    
    return new_X,new_Y

def build_final_condition_derivative(X,Y,K):
    x_min,x_max = np.min(X[:,0]),np.max(X[:,0])
    
    final_condition_points = np.vstack([np.linspace(x_min,x_max,100),np.ones(100)]).T
    final_condition_values = np.array([1 if x >= K else 0 for x in final_condition_points[:,0]])

    new_X = np.vstack((X,final_condition_points))
    new_Y = np.hstack((Y,final_condition_values))
    
    return final_condition_values,final_condition_points

def build_final_condition_2nd_derivative(X,Y,K):
    x_min,x_max = np.min(X[:,0]),np.max(X[:,0])
    
    final_condition_points = np.vstack([np.linspace(x_min,x_max,100),np.ones(100)]).T
    final_condition_values = np.array([0 for _ in final_condition_points[:,0]])

    new_X = np.vstack((X,final_condition_points))
    new_Y = np.hstack((Y,final_condition_values))
    
    return final_condition_values,final_condition_points

# new_X,new_Y = build_final_condition(S,Y,K_)

def get_discrete_integral_matrix(t, center=True):
    """Generative the discrete integration matrix."""
    n = np.size(t)
    td = t[1] - t[0]
    A = np.tril(td * np.ones((n, n)), 0)
    if center:
        for i in range(n):
            A[i, i] = A[i, i] / 2
        A[:, 0] = A[:, 0] - A[0, 0]
    return (A)

def black_scholes_theta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = -sigma * S * norm.pdf(d1) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    return theta

# Define the partial derivatives for the Black-Scholes formula
def black_scholes_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta

def black_scholes_gamma(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma

def black_scholes(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call

def transform_to_heat_eq(u,S,t,sigma,r):
    x = np.log(S)
    T = np.max(t)
    tau = (sigma**2)/2*(T-t)

    k = 2*r/(sigma**2)

    alpha = -(k - 1)/2
    beta = -(k+1)**2/4

    v = u/np.exp(alpha*x + beta*tau)

    return v, x,tau

def numerical_derivative(values, axis, h):
    # Calculate the numerical derivative using central differences
    if axis == 0:  # derivative with respect to S
        deriv = (np.roll(values, -1, axis=0) - np.roll(values, 1, axis=0)) / (2 * h)
    else:  # derivative with respect to T
        deriv = (np.roll(values, -1, axis=1) - np.roll(values, 1, axis=1)) / (2 * h)
    # Fix the boundaries (one-sided differences)
    if axis == 0:
        deriv[0, :] = (values[1, :] - values[0, :]) / h
        deriv[-1, :] = (values[-1, :] - values[-2, :]) / h
    else:
        deriv[:, 0] = (values[:, 1] - values[:, 0]) / h
        deriv[:, -1] = (values[:, -1] - values[:, -2]) / h
    return deriv

def numerical_second_derivative(values, axis, h):
    # Calculate the numerical second derivative using central differences
    if axis == 0:  # second derivative with respect to S
        deriv = (np.roll(values, -1, axis=0) - 2 * values + np.roll(values, 1, axis=0)) / (h**2)
    else:  # second derivative with respect to T
        deriv = (np.roll(values, -1, axis=1) - 2 * values + np.roll(values, 1, axis=1)) / (h**2)
    # Fix the boundaries (use one-sided differences)
    if axis == 0:
        deriv[0, :] = (values[2, :] - 2 * values[1, :] + values[0, :]) / (h**2)
        deriv[-1, :] = (values[-1, :] - 2 * values[-2, :] + values[-3, :]) / (h**2)
    else:
        deriv[:, 0] = (values[:, 2] - 2 * values[:, 1] + values[:, 0]) / (h**2)
        deriv[:, -1] = (values[:, -1] - 2 * values[:, -2] + values[:, -3]) / (h**2)
    return deriv



def numerical_partial_black_scholes(V,r,sigma):
    '''
    Numerical calculation of the black-scholes through DURA-MOS¸NEAGU
    '''
    N = len(V[0])
    M = len(V[:,0])
    V_dot = np.zeros([M,N])

    alpha = 0.5*(sigma**2)
    beta = 0.5*r

    #Calculate option-value
    for m in range(M):
        v_m = []
        for n in range(1,N-1):
                v_m.append((beta*n - alpha*(n**2))*V[m,n-1] + 2*(beta + alpha*(n**2))*V[m,n] - (alpha*(n**2) + beta*n)*V[m,n+1])
        V_dot[m] = [0] + v_m + [0]
    return V_dot


def remove_duplicates_recursive(arr):

    unique_elements = []
    for item in arr:
        if isinstance(item,np.ndarray):
            unique_elements.append(remove_duplicates_recursive(item))
        elif item not in unique_elements:
            unique_elements.append(item)

    return unique_elements

