import cv2
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim

def ssim_compare(img1_path, img2_path) :
    img1 = cv2.imread(img1_path, 0)
    img2 = cv2.imread(img2_path, 0)
    ssim_score, dif = ssim(img1, img2, full=True)
    return ssim_score

ssim_val = ssim_compare('./moon.png', './output_images/laplacian-convolution/sharpened_image.png')
ssim_val1 = ssim_compare('./moon.png', './output_images/unsharp/sharpened_image.png')
print(ssim_val)
print(ssim_val1)