import cv2
import numpy as np
import os

# Create a new directory to save images
output_dir = './output_images/unsharp'
os.makedirs(output_dir, exist_ok=True)

# Read the input image in grayscale
f = cv2.imread('./moon.png', 0)
f = f / 255

# Save and display the original image
cv2.imwrite(os.path.join(output_dir, 'original_image.png'), f * 255)
cv2.imshow('Original Image', f)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Blur the image using GaussianBlur
f_blur = cv2.GaussianBlur(src=f, ksize=(31,31), sigmaX=0, sigmaY=0)

# Save and display the blurred image
cv2.imwrite(os.path.join(output_dir, 'blurred_image.png'), f_blur * 255)
cv2.imshow('Blurred Image', f_blur)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create the mask by subtracting the blurred image from the original image
g_mask = f - f_blur

# Save and display the mask
cv2.imwrite(os.path.join(output_dir, 'mask_image.png'), g_mask * 255)
cv2.imshow('Mask', g_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply unsharp masking
k = 1
g = f + k * g_mask
g = np.clip(g, 0, 1)

# Save and display the final sharpened image
cv2.imwrite(os.path.join(output_dir, 'sharpened_image.png'), g * 255)
cv2.imshow('Sharpened Image', g)
cv2.waitKey(0)
cv2.destroyAllWindows()
