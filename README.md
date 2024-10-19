# Canny Edge Detection with Custom Implementation

## Overview
This project is an implementation of the Canny Edge Detection algorithm, one of the most popular techniques for detecting edges in images. Instead of relying solely on OpenCV's built-in `cv2.Canny` function, we’ve broken down the steps involved and implemented each phase from scratch using Python and NumPy, providing a deeper understanding of how edge detection works. Additionally, the code supports real-time edge detection through a webcam feed.

The project is modularized, making it easier to understand and extend. Each key operation, such as grayscale conversion, Gaussian filtering, Sobel filtering, non-maximum suppression, and hysteresis, is defined in separate functions, spread across different files for clarity.

![showcase](https://github.com/user-attachments/assets/2a079c69-fb27-409b-a1d2-7478e37d53da)

## Features
- **Custom Canny Edge Detection**: Complete step-by-step breakdown of the Canny Edge Detection algorithm.
- **Real-time Processing**: Uses your computer's webcam to perform live edge detection.
- **Modularized Code**: Each part of the edge detection pipeline is implemented in a different file, making the code clean and easy to understand.
- **Efficient Performance**: Uses efficient techniques like OpenCV’s `filter2D` for convolution and OpenCV’s `dilate` for hysteresis tracking.

## Algorithm Breakdown
The Canny Edge Detection process is divided into six main steps:
1. **Grayscale Conversion**: Convert the input image from color to grayscale.
2. **Gaussian Smoothing**: Apply a Gaussian filter to smooth the image and reduce noise.
3. **Gradient Calculation**: Use Sobel filters to compute the gradient magnitude and direction of the image.
4. **Non-Maximum Suppression**: Thin out the edges by suppressing non-maximum pixels in the gradient direction.
5. **Double Thresholding**: Classify edges as strong, weak, or non-edges using two thresholds.
6. **Hysteresis**: Trace edges by connecting weak edges that are connected to strong edges.

These steps are implemented in separate files, which are imported into the main script for easy execution and extension.

## File Structure
Here’s a quick look at how the project is organized:
```
📂 canny_edge_detection/
├── 📂 src/
│   ├── grayscale.py           # Converts RGB to grayscale
│   ├── gaussian.py            # Applies Gaussian blur using a custom kernel
│   ├── sobel.py               # Calculates gradients using Sobel operators
│   ├── nms.py                 # Performs Non-Maximum Suppression
│   ├── threshold.py           # Applies double thresholding
│   └── hysteresis.py          # Performs edge tracking by hysteresis
├── main.py                    # Main script for real-time edge detection
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## Installation

### Requirements
- Python 3.x
- OpenCV
- NumPy

To install the required packages, you can use:
```bash
pip install -r requirements.txt
```

### Running the Project
Once you have all dependencies installed, you can run the project using:

```bash
python main.py
```

This will open a window displaying your webcam feed with real-time edge detection applied. Press 'q' to quit the program.

## Code Explanation
### Grayscale Conversion
The function `rgb_to_grayscale()` converts the input image from RGB to grayscale using the luminosity method. This reduces the image to a 2D array, making it simpler to process.

```python
def rgb_to_grayscale(image):
    return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
```

### Gaussian Smoothing
The `gaussian_kernel()` function generates a Gaussian kernel matrix based on the size and sigma value provided. This kernel is then convolved with the grayscale image to smooth it and reduce noise.

```python
def gaussian_kernel(size, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    return kernel / np.sum(kernel)
```

### Gradient Calculation
Gradients in both the x and y directions are computed using Sobel filters, which help in detecting the edges by highlighting areas with high intensity changes.

```python
def sobel_filters(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    
    Ix = apply_convolution(image, Kx)
    Iy = apply_convolution(image, Ky)
    
    G = np.hypot(Ix, Iy)
    G = (G / G.max()) * 255
    
    return G.astype(np.uint8), np.arctan2(Iy, Ix)
```

### Non-Maximum Suppression
The `non_maximum_suppression()` function ensures that only the local maxima in the gradient direction are preserved, helping to thin out the edges.

```python
def non_maximum_suppression(G, theta):
    M, N = G.shape
    Z = np.zeros((M, N), dtype=np.uint8)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180
    ...
```

### Double Thresholding
This step classifies edges as strong, weak, or non-edges based on two thresholds: high and low. Pixels with gradient magnitudes above the high threshold are considered strong, while those between the high and low thresholds are weak edges.

```python
def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.15):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    ...
```

### Hysteresis
The final step in the Canny process is edge tracking by hysteresis. Weak edges that are connected to strong edges are converted to strong edges, while the rest are discarded.

```python
def hysteresis(img, weak, strong=255):
    kernel = np.ones((3,3), np.uint8)
    while True:
        dilated = cv2.dilate(res, kernel)
        ...
```

## Contributing
Feel free to fork this project and open a pull request if you'd like to contribute! Whether it’s adding new features, fixing bugs, or optimizing the performance, contributions are welcome.

## License
This project is licensed under the MIT License.
