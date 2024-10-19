import numpy as np

def rgb_to_grayscale(image):
    if image.ndim == 3:
        grayscale = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        return grayscale
    elif image.ndim == 2:
        return image
    else:
        raise ValueError(f"Unsupported image dimensions: {image.ndim}")
