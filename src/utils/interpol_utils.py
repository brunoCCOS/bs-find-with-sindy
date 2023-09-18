import numpy as np
from scipy import interpolate
import statsmodels.api as sm

# from pykrige.ok import OrdinaryKriging

def polynomial_interpolation(u, x, y, a, b, degree=3):
    # Fit a polynomial to the data
    coefficients = np.polyfit(x, y, degree)
    polynomial = np.poly1d(coefficients)

    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the option prices using the polynomial
    zi = polynomial(xi, yi)
    
    return zi

def linear_interpolation(u, x, y, a, b):
    f = interpolate.interp2d(x, y, u, kind='linear')
    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    return f(xi, yi)

def cubic_spline_interpolation(u, x, y, a, b):
    f = interpolate.interp2d(x, y, u, kind='cubic')
    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    return f(xi, yi)

def bilinear_interpolation(u, x, y, a, b):
    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    xi, yi = np.meshgrid(xi, yi)
    return interpolate.griddata((x, y), u, (xi, yi), method='linear')

def bicubic_interpolation(u, x, y, a, b):
    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    xi, yi = np.meshgrid(xi, yi)
    print(xi.shape)
    print(yi.shape)s
    return interpolate.griddata((x, y), u, (xi, yi), method='cubic')

def thin_plate_splines_interpolation(u, x, y, a, b):
    rbf = interpolate.Rbf(x, y, u, function='thin_plate')
    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    xi, yi = np.meshgrid(xi, yi)
    return rbf(xi, yi)

# def kriging_interpolation(u, x, y, a, b):
#     # Create an OrdinaryKriging object
#     OK = OrdinaryKriging(x, y, u, variogram_model='linear', verbose=False)

#     xi = np.linspace(min(x), max(x), a)
#     yi = np.linspace(min(y), max(y), b)
#     xi, yi = np.meshgrid(xi, yi)

#     # Interpolate the option prices
#     zi, _ = OK.execute('grid', xi.ravel(), yi.ravel())
#     zi = zi.reshape(xi.shape)
    
#     return zi

def loess_interpolation(u, x, y, a, b):
    # Create a lowess object
    lowess = sm.nonparametric.lowess

    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the option prices
    zi = lowess(u, np.column_stack((x, y)), frac=0.5)
    zi = interpolate.griddata(np.column_stack((x, y)), zi[:, 1], (xi, yi), method='linear')
    
    return zi

def rbf_interpolation(u, x, y, a, b):
    xi = np.linspace(min(x), max(x), a)
    yi = np.linspace(min(y), max(y), b)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate the option prices using RBF
    rbf = interpolate.Rbf(x, y, u)
    zi = rbf(xi, yi)

    return zi