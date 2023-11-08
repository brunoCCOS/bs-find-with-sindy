

import numpy as np
from scipy import interpolate
from pykrige.ok import OrdinaryKriging
from utils.optimization_utils import *
from utils.data_utils import *
from scipy.spatial.distance import cdist
from scipy.interpolate import SmoothBivariateSpline


def smooth_spline_interpolation(points, values):
    # Ensure that the input is a numpy array
    points = np.array(points)
    values = np.array(values)

    # Extract the x and y coordinates
    x = points[:, 0]
    y = points[:, 1]
    
    # Create the spline interpolator
    spline = SmoothBivariateSpline(x, y, values)
    
    # Function to compute the surface and derivatives at a given point
    def compute_at_point(x, y):
        surface = spline(x, y)
        derivative_x = spline(x, y, dx=1)
        derivative_y = spline(x, y, dy=1)
        derivative_xx = spline(x, y, dx=2)
        derivative_yy = spline(x, y, dy=2)
        return surface, derivative_x, derivative_y, derivative_xx, derivative_yy
    
    return compute_at_point

def RBFN_2d(X, Y, rbf_class, epsilon=1e-6):
    n = Y.shape[0]

    # Compute the distance matrix between all pairs of center points
    dist_matrix = cdist(X, X)

    # Add epsilon to the diagonal for numerical stability
    np.fill_diagonal(dist_matrix, dist_matrix.diagonal() + epsilon)

    # Compute the RBF matrix
    RBF_matrix = rbf_class.eval_func(dist_matrix)

    # Solve for the coefficients using linear least squares
    coefficients, _, _, _ = np.linalg.lstsq(RBF_matrix, Y)

    def compute_at_point(x, y):
        x_grid, y_grid = np.meshgrid(x, y)
        grid_points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        m = grid_points.shape[0]
        interpolated_values = np.zeros(m)
        interpolated_values_x = np.zeros(m)
        interpolated_values_xx = np.zeros(m)
        interpolated_values_y = np.zeros(m)

        for i in range(m):
            # Compute distances from the current grid point to all centers
            distances = X - grid_points[i]
            distances[distances == 0] = epsilon
            # Evaluate the radial basis function for these distances
            g_i = rbf_class.eval_func(np.linalg.norm(distances,axis=1))
            interpolated_values[i] = np.dot(coefficients, g_i)

            # Compute first and second derivatives for x
            h_i_x = rbf_class.eval_func_derivative_2d(np.atleast_2d(distances), axis=1)
            H_i_xx = rbf_class.eval_func_2_derivative_2d(np.atleast_2d(distances), axis=1)
            interpolated_values_x[i] = np.dot(coefficients, h_i_x)
            interpolated_values_xx[i] = np.dot(coefficients, H_i_xx)

            # Compute first and second derivatives for y
            h_i_y = rbf_class.eval_func_derivative_2d(np.atleast_2d(distances), axis=2)
            if grid_points[i,1] != 0:
                interpolated_values_y[i] = np.dot(coefficients, h_i_y)
            # interpolated_values_yy[i] = np.dot(coefficients, H_i_yy)  # If needed

        return interpolated_values, interpolated_values_x, interpolated_values_xx, interpolated_values_y, coefficients

    return compute_at_point



def IRBFN2_1d(X,Y,rbf_class,epsilon = 1e-6):
    '''
    Function for Indirect Radial basis function network
    args:
    X: set of center points
    Y: target function evaluated at X
    g: radial basis function
    h: Primitive of the rbf
    H: Primitive of h
    epsilon: staiblity control term
    '''
    # Create a grid of points for interpolation
    x_max, y_max = np.max(X), np.max(Y)

    x_min = np.min(X)
    x_grid= np.linspace(x_min, x_max, 100)

    n = Y.shape[0]
    m = x_grid.shape[0]

    X_2d = np.hstack((X.reshape(-1,1),np.zeros((n,1))))
    dist_matrix = cdist(X_2d,X_2d)
    
    # Add epsilon to the diagonal of the distance matrix to avoid singularities
    np.fill_diagonal(dist_matrix, epsilon)

    # Compute the RBF matrix
    RBF_matrix = rbf_class.eval_func_2_int_1d(dist_matrix)
    
    #Add columns for constant C1 and C2
    constant_matrix = np.hstack((X.reshape(-1,1),np.ones(n).reshape(-1,1)))


    full_matrix = np.hstack((RBF_matrix,constant_matrix))
    
    # Solve for the coefficients using linear equations (Ax = Y)
    coefficients = solve_SVD_system(full_matrix, Y)

    # Interpolate values for the grid points
    interpolated_values = np.zeros(m)
    # Interpolate values for the first derivative grid points
    interpolated_values_x = np.zeros(m)
    # Interpolate values for the seconde derivative grid points
    interpolated_values_xx = np.zeros(m)

    for i in range(m):
        distances = np.abs(X - x_grid[i]) + epsilon
        H_i = np.hstack((rbf_class.eval_func_2_int_1d(distances),x_grid[i],1))
        h_i = np.hstack((rbf_class.eval_func_int_1d(distances),1))
        g_i = rbf_class.eval_func(distances)
        interpolated_values[i] = np.sum(coefficients * H_i)
        interpolated_values_x[i] = np.sum(coefficients[:-1] * h_i)
        interpolated_values_xx[i] = np.sum(coefficients[:-2] * g_i)

    return x_grid, interpolated_values,interpolated_values_x,interpolated_values_xx,coefficients


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


