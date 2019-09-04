import cv2
import numpy as np
import pywt

def wavelet_transform(img, name):
    LL, (LH, HL, HH) = pywt.dwt2(img, 'bior1.1')
    LH *= 20
    HL *= 20
    HH *= 40
    convas = np.concatenate((np.concatenate((LL, LH), axis=1), np.concatenate((HL, HH), axis=1)), axis=0)
    cv2.imwrite('exp/' + name + '.jpg', convas)
    return convas
        
    
img = cv2.imread('results/0_0.jpg',  cv2.IMREAD_GRAYSCALE)

ori_img, previous_img, img, _, _ = np.split(img, 5, axis=1)
#wavelet_transform(ori_img, 'ori')
#wavelet_transform(previous_img, 'pre')

from skimage.measure import compare_ssim
print(compare_ssim(ori_img, img))
#for i in range(3, 15, 2):
#    print(i, compare_ssim(ori_img, img, win_size=i))

_,  S = compare_ssim(ori_img, img, full=True)
print(_, S.shape, np.max(S), np.min(S))
cv2.imwrite('ssim.jpg', S * 255)
_,  S = compare_ssim(ori_img, previous_img, full=True)
print(S.shape, np.max(S), np.min(S))
cv2.imwrite('ssim_previous.jpg', S * 255)

#_,  G = compare_ssim(ori_img, img, gradient=True)
#S = G
#print(_, S.shape, np.max(S), np.min(S))
#cv2.imwrite('ssim_gradient.jpg', S * 255)

_,  S = compare_ssim(ori_img, img, full=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
print(_, S.shape, np.max(S), np.min(S))
cv2.imwrite('ssim_gaussian.jpg', S * 255)
