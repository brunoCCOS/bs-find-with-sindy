
import numpy as np

def print_model(coefs:list,var_names:list,precision = 2):
    eq = 'u_t ='
    for coef,var in zip(coefs, var_names):
        if coef != 0:
            eq += ' + ' + f"{round(coef,precision)}" + ' ' + var
    print(eq)