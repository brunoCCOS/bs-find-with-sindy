import numpy as np

class multiquadratic_rbf:

    def eval_func(self,r):
        return np.sqrt(r**2 + 1)

    def eval_func_int_1d(self,r):
        return (r*self.eval_func(r))/2+ 1/2*np.log(r + self.eval_func(r))
    
    def eval_func_2_int_2d(self,r):
        r1 = -r[:,0]
        r2 = -r[:,1]
        return (r1**2 + r2**2 + 1)**(1.5)/6 + (r1**2 + r2**2 + r1**2 + 1)/2*r1*np.log(r1 + np.sqrt(r1**2 + r2**2 + 1))
    
    def eval_func_int_2d(self,r):
        r1 = -r[:,0]
        r2 = -r[:,1]
        return r1*np.sqrt(r1**2 + r2**2 + 1)/2 + (r2**2 +1)/2*np.log(r1 + np.sqrt(r1**2 + r2**2 + 1))

    def eval_func_2_int_1d(self,r):
        return ((r**2) + 1)**1.5/6 + 1/2*r*np.log(r + self.eval_func(r)) - 1/2*self.eval_func(r)

    def eval_func_derivative_2d(self,r):
        r1 = -r[:,0]
        r2 = -r[:,1]
        return r1/np.sqrt(r1**2 + r2**2 + 1)

    def eval_func_2_derivative_2d(self,r):
        r1 = -r[:,0]
        r2 = -r[:,1]
        return (r2**2 + 1)/(r1**2 + r2**2 + 1)**(3/2)
    

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