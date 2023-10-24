import pandas as pd
import numpy as np
from scipy.stats import norm

def load_data(filename):
    df = pd.read_csv(filename)
    return df


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

def black_scholes_call_time_(S,K,T,r,sigma):
    '''
    Generate the values of the black scholes equation differentiate with respect to time
    '''
    d1 = 1/(sigma*np.sqrt(T))*(np.log(S/K) + (r + (sigma**2)/2)*T)
    call_time = - K * np.exp(-r * T) * r * norm.cdf(d1  - sigma * np.sqrt(T)) - S * ((sigma)/(2*np.sqrt(T))) * norm.pdf(d1)
    return call_time

def black_scholes_call(S, X, T, r, sigma):
    '''
    Generate the price of an option through the black scholes equation
    '''
    d1 = (np.log(S / X) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / X) + (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    call_price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    return call_price

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