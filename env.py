import numpy as np
import sys
import cv2
import torch
from skimage.measure import compare_ssim

class Env():
    def __init__(self, move_range=3, reward_method='square'):
        self.move_range = move_range
        self.batch_size = None
        self.ori_image = None
        self.image = None
        self.previous_image = None

        self.reward_method = reward_method 

        self.laplace_param = 0.1
        self.unmask_param = 0.5
        self.sobel_h1_param = 0.1
        self.sobel_h2_param = 0.1
        self.sobel_v1_param = 0.1
        self.sobel_v2_param = 0.1
    
    def reset(self, ori_image, image):
        self.batch_size = ori_image.shape[0]
        self.ori_image = ori_image
        self.image = image
        self.previous_image = None

    def set_param(self, laplace_param=None, unmask_param=None, sobel_h1_param=None, sobel_h2_param=None, sobel_v1_param=None, sobel_v2_param=None):
        if laplace_param is not None:
            self.laplace_param = laplace_param
        if unmask_param is not None:
            self.unmask_param = unmask_param
        if sobel_h1_param is not None:
            self.sobel_h1_param = sobel_h1_param
        if sobel_h2_param is not None:
            self.sobel_h2_param = sobel_h2_param
        if sobel_v1_param is not None:
            self.sobel_v1_param = sobel_v1_param
        if sobel_v2_param is not None:
            self.sobel_v2_param = sobel_v2_param

        #print(self.laplace_param)
        #print(self.unmask_param)
        #print(self.sobel_h1_param)
        #print(self.sobel_h2_param)
        #print(self.sobel_v1_param)
        #print(self.sobel_v2_param)
        return

    def step(self, act):
        self.previous_image = self.image.copy()

#        neutral = (self.move_range - 1)/2
#        move = act.astype(np.float32)
#        move = (move - neutral) / 255
#        moved_image = self.image + move[:,np.newaxis,:,:]
        gaussian = np.zeros(self.image.shape, self.image.dtype)
        #gaussian2 = np.zeros(self.image.shape, self.image.dtype)
        bilateral = np.zeros(self.image.shape, self.image.dtype)
        #bilateral2 = np.zeros(self.image.shape, self.image.dtype)
        median = np.zeros(self.image.shape, self.image.dtype)
        box = np.zeros(self.image.shape, self.image.dtype)

        laplace5 = np.zeros(self.image.shape, self.image.dtype)
        unmask = np.zeros(self.image.shape, self.image.dtype)
        sobel_h1 = np.zeros(self.image.shape, self.image.dtype)
        sobel_h2 = np.zeros(self.image.shape, self.image.dtype)
        sobel_v1 = np.zeros(self.image.shape, self.image.dtype)
        sobel_v2 = np.zeros(self.image.shape, self.image.dtype)
        b, c, h, w = self.image.shape
        for i in range(self.batch_size):
            if True:#np.sum(act[i]==self.move_range) > 0:
                gaussian[i, 0] = cv2.GaussianBlur(self.image[i,0], ksize=(5,5), sigmaX=0.5)
            if np.sum(act[i]==self.move_range+1) > 0:
                bilateral[i, 0] = cv2.bilateralFilter(self.image[i,0], d=5, sigmaColor=0.1, sigmaSpace=5)
            if np.sum(act[i]==self.move_range+2) > 0:
                median[i, 0] = cv2.medianBlur(self.image[i,0], ksize=5)
            if np.sum(act[i]==self.move_range+3) > 0:
                p = self.laplace_param[i]
                k5 = np.array([[0, -p, 0], [-p, 1 + 4 * p, -p], [0, -p, 0]])
                laplace5[i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k5)
            if np.sum(act[i]==self.move_range+4) > 0:
                amount = self.unmask_param[i]
                unmask[i, 0] = self.image[i, 0] * (1 + amount) - gaussian[i, 0] * amount

            if np.sum(act[i]==self.move_range+5) > 0:
                box[i, 0] = cv2.boxFilter(self.image[i,0], ddepth=-1, ksize=(5,5))

            if np.sum(act[i]==self.move_range+6) > 0:
                p = self.sobel_h1_param[i]
                k = np.array([[-p,-2 * p,-p], [0, 1, 0], [p, 2 * p, p]])
                sobel_h1[i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)
            if np.sum(act[i]==self.move_range+7) > 0:
                p = self.sobel_h2_param[i]
                k = np.array([[p, 2 * p, p], [0, 1, 0], [-p, -2 * p, -p]])
                sobel_h2[i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)
            if np.sum(act[i]==self.move_range+8) > 0:
                p = self.sobel_v1_param[i]
                k = np.array([[p, 0, -p], [2 * p, 1, -2 * p], [p, 0, -p]])
                sobel_v1[i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)
            if np.sum(act[i]==self.move_range+9) > 0:
                p = self.sobel_v2_param[i]
                k = np.array([[-p, 0, p], [-2 * p, 1, 2 * p], [-p, 0, p]])
                sobel_v2[i, 0] = cv2.filter2D(self.image[i, 0], -1, kernel=k)
        
        self.image = np.where(act[:,np.newaxis,:,:]==1, self.image + 1 / 255, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==2, self.image - 3 / 255, self.image)


        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range, gaussian, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+1, bilateral, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+2, median, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+3, laplace5, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+4, unmask, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+5, box, self.image)

        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+6, sobel_h1, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+7, sobel_h2, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+8, sobel_v1, self.image)
        self.image = np.where(act[:,np.newaxis,:,:]==self.move_range+9, sobel_v2, self.image)

        self.image = np.clip(self.image, 0, 1) # 0828
        if self.reward_method == 'square':
            reward = np.square(self.ori_image - self.previous_image) * 255 - np.square(self.ori_image - self.image) * 255
        elif self.reward_method == 'abs':
            reward = np.abs(self.ori_image - self.previous_image) * 255 - np.abs(self.ori_image - self.image) * 255
        elif self.reward_method == 'ssim':
            reward = np.abs(self.ori_image - self.previous_image) * 255 - np.abs(self.ori_image - self.image) * 255
            for ii in range(self.image.shape[0]):
                reward[ii, 0] +=  (- compare_ssim(self.ori_image[ii, 0] * 255, self.previous_image[ii, 0] * 255, full=True)[1] + compare_ssim(self.ori_image[ii, 0] * 255, self.image[ii, 0] * 255, full=True)[1]) * 5
        return self.image, reward 

