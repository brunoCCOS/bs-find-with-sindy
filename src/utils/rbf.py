import numpy as np
from scipy.special import erf
class multiquadratic_rbf:

    def __init__(self, c=1.0):
        self.c = c

    def eval_func(self, r):
        return np.sqrt(r**2 + self.c**2)

    def eval_func_derivative_2d(self, r, axis):
        r1 = r[:,0]
        r2 = r[:,1]
        r_norm = np.sqrt(r1**2 + r2**2 + self.c**2)
        if axis == 0:
            return r1 / r_norm
        else:
            return r2 / r_norm

    def eval_func_2_derivative_2d(self, r, axis):
        r1 = r[:,0]
        r2 = r[:,1]
        r_norm = np.sqrt(r1**2 + r2**2 + self.c**2)
        if axis == 0:
            return (r2**2 + self.c**2) / r_norm**3
        else:
            return (r1**2 + self.c**2) / r_norm**3

class quintic_rbf:

    def eval_func(self, r):
        return r**5

    def eval_func_derivative_2d(self, r, axis):
        # First derivative in 2D of the quintic RBF
        r1 = r[:,0]
        r2 = r[:,1]
        r_norm = np.sqrt(r1**2 + r2**2)
        if axis == 0:
            return 5 * r1 * r_norm**3
        else: 
            return 5 * r2 * r_norm**3

    def eval_func_2_derivative_2d(self, r, axis):
        # Second derivative in 2D of the quintic RBF
        r1 = r[:,0]
        r2 = r[:,1]
        r_norm = np.sqrt(r1**2 + r2**2)
        if axis == 0:
            return 5 * r_norm**3 + 15 * r1**2 * r_norm
        else:
            return 5 * r_norm**3 + 15 * r2**2 * r_norm
class inverse_multiquadratic_rbf:

    def __init__(self, c=1.0):
        self.c = c

    def eval_func(self, r):
        return 1.0 / np.sqrt(r**2 + self.c**2)

    def eval_func_derivative_2d(self, r, axis):
        r1 = r[:,0]
        r2 = r[:,1]
        r_norm = np.sqrt(r1**2 + r2**2 + self.c**2)
        if axis == 0:
            return -r1 / r_norm**3
        else:
            return -r2 / r_norm**3

    def eval_func_2_derivative_2d(self, r, axis):
        r1 = r[:,0]
        r2 = r[:,1]
        r_norm = np.sqrt(r1**2 + r2**2 + self.c**2)
        if axis == 0:
            return (3 * r1**2 - r2**2 - self.c**2) / r_norm**5
        else:
            return (3 * r2**2 - r1**2 - self.c**2) / r_norm**5

class cubic_rbf:

    def eval_func(self,r):
        return r**3

    def eval_func_int_1d(self,r):
        return (r*self.eval_func(r))/2+ 1/2*np.log(r + self.eval_func(r))
    
    def eval_func_2_int_2d(self,r):
        r1 = -r[:,0]
        r2 = -r[:,1]
        return (r**2 + 1)**1.5/6 + (r**2 - (r1)**2 + 1)*r1/2*np.log(-r1+self.eval_func(r)) - (r**2 - (r2)**2 + 1)/2*self.eval_func(r)
    
    def eval_func_int_2d(self,r):
        r1 = -r[:,0]
        r2 = -r[:,1]
        return -1/8*(-r1)*(r1**2/r2**2 + 1)*np.sqrt(r1**2 + r2**2)*r2**2*(2/(r1**2/r2**2 + 1) + 3/(r1**2/r2**2 + 1)**2 + (3*r2*np.arcsinh(r1/r2))/(r1*(r1**2/r2**2 + 1)**5/2))

    def eval_func_2_int_1d(self,r):
        return ((r**2) + 1)**1.5/6 + 1/2*r*np.log(r + self.eval_func(r)) - 1/2*self.eval_func(r)

    def eval_func_derivative_2d(self,r,axis):
        r1 = r[:,0]
        r2 = r[:,1]
        if axis == 1:
            return 3*-r1*np.sqrt(r1**2 + r2**2)
        else: 
            return 3*-r2*np.sqrt(r1**2 + r2**2)

    def eval_func_2_derivative_2d(self,r,axis):
        r1 = r[:,0]
        r2 = r[:,1]
        if axis == 1:
            return 3*(2*(r1)**2 + (r2)**2)/np.sqrt((r1)**2 + (r2)**2)
        else:
            return 3*(2*(r1)**2 + (r2)**2)/np.sqrt((r1)**2 + (r2)**2)
        


class gaussian_rbf:

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def eval_func(self, r):
        return np.exp(-(self.epsilon * r)**2)

    def eval_func_int_1d(self, r):
        # Integral of the Gaussian RBF in 1D from 0 to r
        return (np.sqrt(np.pi) / (2 * self.epsilon)) * erf(self.epsilon * r)
    
    def eval_func_2_int_2d(self, r):
        # Placeholder for the second integral in 2D of the Gaussian RBF
        # Exact expression requires numerical integration
        pass
    
    def eval_func_int_2d(self, r):
        # Placeholder for the integral in 2D of the Gaussian RBF
        # Exact expression requires numerical integration
        pass

    def eval_func_2_int_1d(self, r):
        # Placeholder for the second integral in 1D of the Gaussian RBF
        pass

    def eval_func_derivative_2d(self, r, axis):
        # Derivative in 2D of the Gaussian RBF
        r1 = r[:,0]
        r2 = r[:,1]
        r = np.sqrt(r1**2 + r2**2)
        if axis == 0:
            return -2 * self.epsilon**2 * r1 * self.eval_func(r)
        else: 
            return -2 * self.epsilon**2 * r2 * self.eval_func(r)

    def eval_func_2_derivative_2d(self, r, axis):
        # Second derivative in 2D of the Gaussian RBF
        r1 = r[:,0]
        r2 = r[:,1]
        if axis == 0:
            return 2 * self.epsilon**2 * (2 * self.epsilon**2 * r1**2 - 1) * self.eval_func(np.sqrt(r1**2 + r2**2))
        else:
            return 2 * self.epsilon**2 * (2 * self.epsilon**2 * r2**2 - 1) * self.eval_func(np.sqrt(r1**2 + r2**2))


class thin_plate_spline_rbf:

    def eval_func(self, r):
        # Use a small epsilon to avoid log(0) which is undefined
        epsilon = 1e-15
        r = np.maximum(r, epsilon)
        return r**2 * np.log(r)

    def eval_func_int_1d(self, r):
        # Placeholder for the 1D integral of the TPS RBF
        pass

    def eval_func_2_int_2d(self, r):
        # Placeholder for the second integral in 2D of the TPS RBF
        pass
    
    def eval_func_int_2d(self, r):
        # Placeholder for the integral in 2D of the TPS RBF
        pass

    def eval_func_2_int_1d(self, r):
        # Placeholder for the second integral in 1D of the TPS RBF
        pass

    def eval_func_derivative_2d(self, r, axis):
        # First derivative in 2D of the TPS RBF
        # Avoid division by zero
        epsilon = 1e-15
        r = np.maximum(r, epsilon)
        r1 = r[:,0]
        r2 = r[:,1]
        r_norm = np.sqrt(r1**2 + r2**2)
        if axis == 0:
            return 2 * r1 * np.log(r_norm + epsilon) + r1
        else: 
            return 2 * r2 * np.log(r_norm + epsilon) + r2

    def eval_func_2_derivative_2d(self, r, axis):
        # Second derivative in 2D of the TPS RBF
        # Avoid division by zero
        epsilon = 1e-15
        r = np.maximum(r, epsilon)
        r1 = r[:,0]
        r2 = r[:,1]
        r_norm = np.sqrt(r1**2 + r2**2)
        if axis == 0:
            return 2 * np.log(r_norm + epsilon) + 3
        else:
            return 2 * np.log(r_norm + epsilon) + 3