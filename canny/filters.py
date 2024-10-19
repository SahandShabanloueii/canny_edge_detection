import numpy as np
import cv2

def gaussian_kernel(size=3, sigma=1):
    # Ensure the size is odd
    if size % 2 == 0:
        raise ValueError("Size of the kernel must be an odd number.")

    # Initialize the kernel with zeros
    kernel = [[0 for _ in range(size)] for _ in range(size)]
    
    # Calculate the center position
    center = size // 2

    # Calculate the sum for normalization
    sum_val = 0.0

    # Fill the kernel with Gaussian values
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            exponent = -(x**2 + y**2) / (2 * sigma**2)
            kernel[i][j] = (1 / (2 * np.pi * sigma**2)) * np.exp(exponent)
            sum_val += kernel[i][j]

    # Normalize the kernel so that the sum of all elements is 1
    for i in range(size):
        for j in range(size):
            kernel[i][j] /= sum_val

    # Convert the kernel to a NumPy array for further processing
    kernel = np.array(kernel)
    
    return kernel

def apply_convolution(image, kernel):
    
    # Use OpenCV's filter2D function for efficient convolution
    return cv2.filter2D(image, -1, kernel)

def sobel_filters(image):
    # Define Sobel kernels for x and y directions
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1,  2,  1],
                   [0,  0,  0],
                   [-1, -2, -1]], dtype=np.float32)
    
    # Apply the Sobel kernels to get gradients
    Ix = apply_convolution(image, Kx)
    Iy = apply_convolution(image, Ky)
    
    # Calculate the gradient magnitude
    G = np.hypot(Ix, Iy)
    G = (G / G.max()) * 255  # Normalize to 0-255
    G = G.astype(np.uint8)
    
    # Calculate the gradient direction
    theta = np.arctan2(Iy, Ix)
    
    return G, theta
