from scipy.signal import convolve2d
import numpy as np

def moving_average_2d(array, kernel_size):
    """
    Perform a moving average on a 2D array (n x n) across both axes.

    Parameters:
    - array: Input 2D array (n x n).
    - kernel_size: Size of the moving average window.

    Returns:
    - result: 2D array after applying the moving average.
    """
    if kernel_size <= 0:
        raise ValueError("Kernel size must be a positive integer")

    # Create a kernel for moving average
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)

    # Apply 2D convolution to perform the moving average
    result = convolve2d(array, kernel, mode='same', boundary='wrap')

    return result


def tv_denoise(matrix, lambda_tv=0.1, tol=1e-5, max_iter=100):
    """
    Apply Total Variation denoising to a 2D matrix of numerical derivatives.

    Parameters:
    - matrix: Input 2D matrix (n x m) of numerical derivatives.
    - lambda_tv: Regularization parameter controlling the weight of the TV term.
    - tol: Tolerance for stopping criterion.
    - max_iter: Maximum number of iterations.

    Returns:
    - u: The denoised 2D matrix.
    """
    m, n = matrix.shape
    u = np.copy(matrix)  # Initialize u with the input matrix
    ux = np.zeros_like(u)
    uy = np.zeros_like(u)
    px = np.zeros_like(u)
    py = np.zeros_like(u)
    
    # Helper function to compute the gradient
    def gradient(v):
        grad_x = np.roll(v, -1, axis=1) - v
        grad_y = np.roll(v, -1, axis=0) - v
        return grad_x, grad_y
    
    # Helper function to compute the divergence
    def divergence(px, py):
        div_x = np.roll(px, 1, axis=1) - px
        div_y = np.roll(py, 1, axis=0) - py
        return div_x + div_y
    
    # Iterative gradient descent
    for _ in range(max_iter):
        ux, uy = gradient(u)
        
        # Update dual variables
        px_new = px + (1 / lambda_tv) * ux
        py_new = py + (1 / lambda_tv) * uy
        norm_new = np.maximum(1, np.sqrt(px_new ** 2 + py_new ** 2))
        px = px_new / norm_new
        py = py_new / norm_new
        
        # Update primal variable
        u_old = u
        u = matrix + lambda_tv * divergence(px, py)
        
        # Check for convergence
        error = np.linalg.norm(u - u_old) / np.linalg.norm(u)
        if error < tol:
            break
    
    return u


def wiener_filter(matrix, kernel, K=0.01):
    """
    Apply Wiener filtering to a 2D matrix.

    Parameters:
    - matrix: Input 2D matrix (n x m) to be denoised.
    - kernel: The degradation function (point spread function) of the system.
    - K: Noise-to-signal power ratio. This constant is used to control the trade-off
        between inverse filtering and noise smoothing.

    Returns:
    - result: The denoised 2D matrix.
    """
    # Convert the input matrix and kernel to frequency domain
    matrix_fft = np.fft.fft2(matrix)
    kernel_fft = np.fft.fft2(kernel, s=matrix.shape)
    
    # Create Wiener filter
    kernel_conj_fft = np.conj(kernel_fft)
    numerator = kernel_conj_fft
    denominator = kernel_fft * kernel_conj_fft + K
    wiener_filter = numerator / denominator
    
    # Apply Wiener filter
    result_fft = wiener_filter * matrix_fft
    
    # Convert the result back to spatial domain
    result = np.fft.ifft2(result_fft)
    return np.real(result)