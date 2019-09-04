import cv2
import numpy as np

from dataset import load_mask
from mri_utils import Downsample

def Reward(m, ori_image, previous_image, image):
    reward = np.square(ori_image - previous_image) * 255 - np.square(ori_image - image) * 255
    diff = np.abs(ori_image - previous_image) - np.abs(ori_image - image)
    print(m, np.mean(reward), np.max(reward), np.min(reward))
    print(np.mean(diff)*255, np.mean(diff > 0), np.mean(diff < 0), '\n')

    A = np.clip(image * 255, 0, 255).astype(np.uint8)[..., np.newaxis]
    A = np.tile(A, (1, 1, 3))
    A[..., 0] = (diff > 0) * 80
    A[..., 1] = (diff < 0) * 80
    cv2.imwrite('analysis/'+m+'.jpg', image * 255)
    cv2.imwrite('analysis/'+m+'_diff.jpg', A)
    return

ori_image = cv2.imread('0_113.bmp', cv2.IMREAD_GRAYSCALE)
mask = load_mask()
image, _, _ = Downsample(ori_image, mask)
image = np.round(np.clip(image, 0, 255))
ori_image = ori_image / 255.
image = image / 255.
image = image.astype(np.float32)

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 1)
    #sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    #sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    #sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

#kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
#cv2.imwrite('laplace5.jpg', 255 * (cv2.filter2D(image, -1, kernel)))
#cv2.imwrite('unmask.jpg', 255 * (image * 2 - cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0.5) * 1))
#xxx
#for a in range(10):
#    print(a)
#    #Reward('sharpened', ori_image, image, unsharp_mask(image, sigma=0.5, amount=a / 10, threshold=1/255.))
#    Reward('sharpened', ori_image, image, unsharp_mask(image, sigma=0.5, amount=1, threshold=a/255.))

Reward('original', ori_image, image, ori_image)
kernel = np.array([[-1,-2,-1], [0, 0, 0], [1, 2, 1]])
Reward('sobel1', ori_image, image, cv2.filter2D(image, -1, kernel))
kernel_laplace = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
complicated = (cv2.GaussianBlur(cv2.filter2D(image, -1, kernel), ksize=(5,5), sigmaX=0.5) * cv2.filter2D(image, -1, kernel_laplace)) + image
Reward('complicated', ori_image, image, complicated)
kernel = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
Reward('sobel2', ori_image, image, cv2.filter2D(image, -1, kernel))

Reward('GB', ori_image, image, cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0.5))
Reward('GB1.5', ori_image, image, cv2.GaussianBlur(image, ksize=(5,5), sigmaX=1.5))
Reward('+1', ori_image, image, image + 1 / 255.)
Reward('-1', ori_image, image, image - 1 / 255.)
Reward('+2', ori_image, image, image + 2 / 255.)
Reward('-2', ori_image, image, image - 2 / 255.)
Reward('set0', ori_image, image, 0 * image)
Reward('bi', ori_image, image, cv2.bilateralFilter(image, d=5, sigmaColor=0.1, sigmaSpace=5))
Reward('bi1.0', ori_image, image, cv2.bilateralFilter(image, d=5, sigmaColor=1.0, sigmaSpace=5))
Reward('median', ori_image, image, cv2.medianBlur(image, ksize=5))
Reward('box', ori_image, image, cv2.boxFilter(image, ddepth=-1, ksize=(5,5)))

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
Reward('laplace9', ori_image, image, cv2.filter2D(image, -1, kernel))
kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
Reward('laplace5', ori_image, image, cv2.filter2D(image, -1, kernel))


Reward('unmask1', ori_image, image, 1.5 * image - 0.5 * cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0.5))
Reward('unmask2', ori_image, image, 1.5 * image - 0.5 * cv2.GaussianBlur(image, ksize=(5,5), sigmaX=1.5))
Reward('unmask3', ori_image, image, 1.2 * image - 0.2 * cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0.5))
Reward('unmask4', ori_image, image, 1.2 * image - 0.2 * cv2.GaussianBlur(image, ksize=(5,5), sigmaX=1.5))
Reward('unmask5', ori_image, image, 2.0 * image - 1 * cv2.GaussianBlur(image, ksize=(5,5), sigmaX=0.5))
Reward('unmask6', ori_image, image, 2.0 * image - 1 * cv2.GaussianBlur(image, ksize=(5,5), sigmaX=1.5))

