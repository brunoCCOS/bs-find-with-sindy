"""
Interpolation Methods

This module provides various interpolation methods for estimating 2D functions on a  surface
given input data points of spatial and temporal coordinates and corresponding function values.

Functions:
- polynomial_interpolation(u, x, y, a, b, degree=3):
    Perform polynomial interpolation on the input data.
    
- linear_interpolation(u, x, y, a, b):
    Perform linear interpolation on the input data.
    
- cubic_spline_interpolation(u, x, y, a, b):
    Perform cubic spline interpolation on the input data.
    
- bilinear_interpolation(u, x, y, a, b):
    Perform bilinear interpolation on the input data.
    
- bicubic_interpolation(u, x, y, a, b):
    Perform bicubic interpolation on the input data.
    
- thin_plate_splines_interpolation(u, x, y, a, b):
    Perform thin plate splines interpolation on the input data.
    
- kriging_interpolation(x, y, u, a, b):
    Perform Kriging interpolation on spatial-temporal data.
    
- loess_interpolation(u, x, y, a, b):
    Perform LOESS (Locally Weighted Scatterplot Smoothing) interpolation on the input data.
    
- idw_interpolation(u, x, y, a, b):
    Perform Inverse Distance Weighting (IDW) interpolation on the input data

- rbf_interpolation(u, x, y, a, b):
    Perform Radial Basis Function (RBF) interpolation on the input data.
    
- 
Parameters:
- u (array-like): Values of the function evaluated at (x, y).
- x (array-like): Spatial coordinates.
- y (array-like): Temporal coordinates.
- a (int): Number of points in the spatial grid.
- b (int): Number of points in the temporal grid.
- degree (int, optional): Degree of the polynomial for polynomial interpolation (default is 3).

Returns:
- np.ndarray: Interpolated values on the specified grid.

Note: The choice of interpolation method should be based on the nature of your data and specific requirements.
"""


import numpy as np
from scipy import interpolate
import statsmodels.api as sm
from pykrige.ok import OrdinaryKriging

def polynomial_interpolation(u, x, y, a, b, degree=3):
    # Fit a polynomial to the data
    coefficients = np.polyfit((x,y), u, degree)
    polynomial = np.poly1d(coefficients)


    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the option prices using the polynomial
    zi = polynomial(xi, yi)
    
    return zi,xi,yi

def smooth_spline_interpolation(u,x,y,a,b):
    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    tck = interpolate.bisplrep(x, y, u, s=0)
    znew = interpolate.bisplev(xi, yi, tck)
    return znew.T,xi,yi

def linear_interpolation(u, x, y, a, b):
    f = interpolate.interp2d(x, y, u, kind='linear')
    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    return f(xi, yi),xi,yi

def cubic_spline_interpolation(u, x, y, a, b):
    f = interpolate.interp2d(x, y, u, kind='cubic')
    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    return f(xi, yi),xi,yi

def bilinear_interpolation(u, x, y, a, b):
    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    xi, yi = np.meshgrid(xi, yi)
    return interpolate.griddata((x, y), u, (xi, yi), method='linear'),xi,yi

def bicubic_interpolation(u, x, y, a, b):
    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    xi, yi = np.meshgrid(xi, yi)

    return interpolate.griddata((x, y), u, (xi, yi), method='cubic', rescale=True),xi,yi


def kriging_interpolation(u, x, y, a, b):
    # Create a spatial-temporal grid
    grid_x = np.linspace(min(x), max(x), a)
    grid_y = np.linspace(min(y), max(y), b)
    
    # Create mesh grid for spatial and temporal coordinates
    spatial_mesh, temporal_mesh = np.meshgrid(grid_x, grid_y)
    
    # Flatten the mesh grids to get all spatial-temporal coordinates
    grid_points = np.column_stack((spatial_mesh.ravel(), temporal_mesh.ravel()))
    
    # Create an Ordinary Kriging instance
    ok = OrdinaryKriging(x, y, u, variogram_model='spherical')
    
    # Interpolate values on the grid
    z, _ = ok.execute('grid', grid_x, grid_y)
    
    return z.reshape((b, a)),spatial_mesh,temporal_mesh

def idw_interpolation(u, x, y, a, b, power=2, epsilon=1e-6):
    # Ensure that inputs are NumPy arrays
    u = np.array(u)
    x = np.array(x)
    y = np.array(y)

    if x.shape[0] != len(u) or y.shape[0] != len(u):
        raise ValueError("x, y, and u must have the same length.")

    # Define the range of x and y values for interpolation
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)

    # Generate a grid of target points
    x_target = np.linspace(x_min, x_max, a)
    y_target = np.linspace(y_min, y_max, b)

    # Initialize the interpolated values grid
    interpolated_values = np.zeros((b, a))  # Swap b and a to match meshgrid

    # Perform IDW interpolation for each target point
    for i in range(b):  # Iterate over y (rows)
        for j in range(a):  # Iterate over x (columns)
            target_point = np.array([x_target[j], y_target[i]])  # Swap x and y

            # Calculate distances between measured points and the target point
            distances = np.linalg.norm(np.column_stack((x, y)) - target_point, axis=1)

            # Add epsilon to avoid division by zero
            distances = distances + epsilon

            # Calculate weights based on inverse distance
            weights = 1.0 / (distances ** power)

            # Normalize the weights
            normalized_weights = weights / np.sum(weights)

            # Interpolate the value
            interpolated_values[i, j] = np.sum(u * normalized_weights)

    return interpolated_values,x_target, y_target

def rbf_interpolation(u, x, y, a, b, kernel='cubic'):
    # Create an RBFInterpolator with the cubic kernel (phi(r) = r^3)
    rbf = interpolate.RBFInterpolator(list(zip(x, y)), u, kernel=kernel)

    # Create a grid of points for interpolation with dimensions a x b
    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)

    # yi = np.unique(np.append(yi,y))

    xi, yi = np.meshgrid(xi, yi)

    xy_points = np.column_stack((xi.ravel(), yi.ravel())) # Flatten the grid points
    zi = rbf(xy_points).reshape(xi.shape)  # Reshape the result to match xi shape


    return zi,xi,yi


def calc_error(interpol_u,u):
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

    print('Interpolation error for Original u')
    print('RSE:', rse)
    print('RMSE:', rmse)
    return(rse,rmse)