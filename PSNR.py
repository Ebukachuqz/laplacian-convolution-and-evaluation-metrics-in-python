import cv2
import numpy as np

def calculate_psnr(original, processed):
    # Ensure the images have the same dimensions
    if original.shape != processed.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    mse = np.mean((original - processed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

# Read the original and processed images
original = cv2.imread('./moon.png')
sharpened1 = cv2.imread('./output_images/laplacian-convolution/sharpened_image.png')
sharpened2 = cv2.imread('./output_images/unsharp/sharpened_image.png')

# Convert images to grayscale
original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
sharpened1_gray = cv2.cvtColor(sharpened1, cv2.COLOR_BGR2GRAY)
sharpened2_gray = cv2.cvtColor(sharpened2, cv2.COLOR_BGR2GRAY)

# Calculate PSNR for both sharpened images
psnr1 = calculate_psnr(original_gray, sharpened1_gray)
psnr2 = calculate_psnr(original_gray, sharpened2_gray)

print(f"PSNR for Image Sharpening Technique 1: {psnr1} dB")
print(f"PSNR for Image Sharpening Technique 2: {psnr2} dB")
