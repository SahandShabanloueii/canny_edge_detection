import numpy as np
import cv2

def threshold(img, lowThresholdRatio=0.08, highThresholdRatio=0.15):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    strong = 255
    weak = 25
    
    strong_i, strong_j = np.where(img >= highThreshold)
    weak_i, weak_j = np.where((img >= lowThreshold) & (img < highThreshold))
    
    res = np.zeros_like(img, dtype=np.uint8)
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return res, weak, strong

def hysteresis(img, weak, strong=255):
    res = img.copy()
    kernel = np.ones((3,3), np.uint8)
    
    while True:
        dilated = cv2.dilate(res, kernel)
        new_strong = np.where((res == weak) & (dilated == strong))
        if len(new_strong[0]) == 0:
            break
        res[new_strong] = strong
    
    res[res != strong] = 0
    return res
