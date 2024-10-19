import cv2
import numpy as np
from canny import (
    rgb_to_grayscale,
    gaussian_kernel,
    apply_convolution,
    sobel_filters,
    non_maximum_suppression,
    threshold,
    hysteresis,
    stack_images
)

def canny_edge_detection(image, 
                         gaussian_kernel_size=3, 
                         gaussian_sigma=5.0, 
                         lowThresholdRatio=0.20, 
                         highThresholdRatio=0.25):
    gray = rgb_to_grayscale(image)
    if gray.ndim != 2:
        raise ValueError(f"Grayscale image should be 2D, but got {gray.ndim}D.")
    
    kernel = gaussian_kernel(gaussian_kernel_size, gaussian_sigma)
    blurred = apply_convolution(gray, kernel)
    
    gradient_magnitude, gradient_direction = sobel_filters(blurred)
    
    non_max = non_maximum_suppression(gradient_magnitude, gradient_direction)
    
    thresholded, weak, strong = threshold(non_max, lowThresholdRatio, highThresholdRatio)
    
    edges = hysteresis(thresholded, weak, strong)
    
    return edges

def main():
    SCALE = 3
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    print("Press 'q' to exit the video stream.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab a frame.")
            break
        
        try:
            frame_flipped = cv2.flip(frame, 1)
            edges = canny_edge_detection(image=frame_flipped)
            combined = stack_images(rgb_to_grayscale(frame_flipped), edges, scale=SCALE)
            cv2.imshow('Canny Edge Detection', combined)
        
        except Exception as e:
            print(f"An error occurred: {e}")
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting the video stream.")
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
