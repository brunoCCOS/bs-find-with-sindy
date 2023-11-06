
import numpy as np

def print_model(coefs:list,var_names:list,precision = 2):
    eq = 'u_t ='
    for coef,var in zip(coefs, var_names):
        if coef != 0:
            eq += ' + ' + f"{round(coef,precision)}" + ' ' + var
    print(eq)

    
def relative_squared_error(y_true, y_pred):
    """
    Calculate the Relative Squared Error (RSE) between true values (y_true) and predicted values (y_pred),
    treating NaN values and zeros in y_true appropriately.

    Parameters:
    y_true -- array of true values
    y_pred -- array of predicted values

    Returns:
    rse -- calculated relative squared error
    """
    epsilon = 1e-2
    # Handle cases where y_true is zero to avoid division by zero
    # Set the relative error to zero in these cases as contribution to the error is considered negligible
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_errors = np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
        # Mask out NaNs and infinities that may have resulted from division by zero
        # mask = ~np.isnan(relative_errors) & ~np.isinf(relative_errors)
        rse =relative_errors
    
    return rse

def root_mean_squared_error(y_true, y_pred):
    """
    Calculate the Root Mean Squared Error (RMSE) between true values (y_true) and predicted values (y_pred),
    treating NaN values appropriately.

    Parameters:
    y_true -- array of true values
    y_pred -- array of predicted values

    Returns:
    rmse -- calculated root mean squared error
    """
    # Handle NaN values by using np.nanmean, which ignores NaNs in the calculation
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    
    return rmse

def calc_error_2d(interpol_u,u,show=False):
    # Calculate the squared difference between interpol_u.T and u while handling NaN values
    squared_diff = (interpol_u - u.reshape((u.shape[0], u.shape[1])))**2

    # Calculate the squared difference between interpol_u.T and u while handling NaN values
    squared_diff_mean = (interpol_u - u.mean())**2

    # Replace NaN values with 0 in the squared difference matrix
    squared_diff[np.isnan(squared_diff)] = 0

    # Replace NaN values with 0 in the squared difference matrix
    squared_diff_mean[np.isnan(squared_diff_mean)] = 0


    # Calculate RSE
    rse = np.sum(squared_diff) / np.sum(squared_diff_mean)

    # Calculate RMSE
    rmse = np.sqrt(np.mean(np.sum(squared_diff, axis=1)))
    if show:
        print('RSE:', rse)
        print('RMSE:', rmse)
    return(rse,rmse)


def calc_error_1d(interpol_u,u,show=False):
    # Calculate the squared difference between interpol_u.T and u while handling NaN values
    mse = np.mean((interpol_u - u)**2)

    rmse = np.sqrt(mse)
    if show:
        print('MSE:', mse)
        print('RMSE:', rmse)
    return(mse,rmse)