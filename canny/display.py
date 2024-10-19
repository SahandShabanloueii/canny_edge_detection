import numpy as np
import cv2

def stack_images(original, edges, scale=1.0):
    if original.max() != 0:
        original_normalized = (original / original.max()) * 255
    else:
        original_normalized = original
    original_normalized = original_normalized.astype(np.uint8)
    
    original_display = cv2.cvtColor(original_normalized, cv2.COLOR_GRAY2BGR)
    edges_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    combined = np.hstack((original_display, edges_display))
    
    if scale != 1.0:
        width = int(combined.shape[1] * scale)
        height = int(combined.shape[0] * scale)
        combined = cv2.resize(combined, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return combined
