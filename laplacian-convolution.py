import cv2
import numpy as np
import os

def convolve2D(image, kernel):
    # Get dimensions of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Calculate padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Initialize output image
    output_image = np.zeros_like(image, dtype=np.float32)

    # Pad the image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='reflect')

    # Perform 2D convolution
    for y in range(image_height):
        for x in range(image_width):
            # Extract the region of interest
            roi = padded_image[y:y+kernel_height, x:x+kernel_width]
            # Element-wise multiplication and summation
            output_image[y, x] = np.sum(roi * kernel)

    return output_image

# Create a new directory to save images
output_dir = './output_images/laplacian-convolution'
os.makedirs(output_dir, exist_ok=True)

# Load the input image
input_image = cv2.imread('./moon.png')

# Convert the image to grayscale
gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Define the Laplacian filter kernel
laplacian_kernel = np.array([[1, 1, 1],
                             [1, -8, 1],
                             [1, 1, 1]], dtype=np.float32)

# Perform convolution
laplacian_filtered = convolve2D(gray_image, laplacian_kernel)

# Multiply Laplacian-filtered image by a constant (-1)
laplacian_filtered *= -1

# Add Laplacian-filtered image to original grayscale image
sharpened_image = cv2.add(gray_image, laplacian_filtered, dtype=cv2.CV_8UC1)

# Save the images
cv2.imwrite(os.path.join(output_dir, 'input_gray_image.png'), gray_image)
cv2.imwrite(os.path.join(output_dir, 'sharpened_image.png'), sharpened_image)
cv2.imwrite(os.path.join(output_dir, 'laplacian_filtered_image.png'), laplacian_filtered)

# Display the images
cv2.imshow("Input Gray Image", gray_image)
cv2.imshow("Sharpened Image", sharpened_image)
cv2.imshow("Laplacian-filtered Image", laplacian_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()
