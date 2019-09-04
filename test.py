import torch
import numpy as np
import cv2
from torch.autograd import Variable

from env import Env
from dataset import load_mask
from pixel_wise_a2c import PixelWiseA2C
from mri_utils import Downsample
from model import MyFcn
from train import computePSNR, computeSSIM

def test_image(model, a2c):
    move_range = 3
    env = Env(move_range)

    avg_reward = 0
    PSNR_list = []
    SSIM_list = []

    ori_image = cv2.imread('test.bmp', cv2.IMREAD_GRAYSCALE)
    mask = load_mask()
    image, _, _ = Downsample(ori_image, mask)
    image = np.round(np.clip(image, 0, 255))

    ori_image = ori_image / 255.
    image = image / 255.

    ori_image = ori_image[np.newaxis, np.newaxis, ...].astype('float32')
    image = image[np.newaxis, np.newaxis, ...].astype('float32')
    previous_image = image.copy()

    env.reset(ori_image=ori_image, image=image)


    for j in range(2):
        image_input = Variable(torch.from_numpy(image).float().cuda(), volatile=True)
        pout, vout = model(image_input)
        actions = a2c.act(pout, deterministic=False)
        image, reward = env.step(actions)
        image = np.clip(image, 0, 1)

        avg_reward += np.mean(reward)


    PSNR_list.append(computePSNR(ori_image[0, 0], previous_image[0, 0], image[0, 0]))
    SSIM_list.append(computeSSIM(ori_image[0, 0], previous_image[0, 0], image[0, 0]))

    actions = actions.astype(np.uint8)
    total = actions.size
    a0 = actions[0]
    B = image[0, 0].copy()
    for a in range(13):
        print(a, 'actions', np.sum(actions==a) / total)
        A = np.zeros((*B.shape, 3))
        A[..., 0] += B * 255
        A[..., 1] += B * 255
        A[..., 2] += B * 255
        A[a0==a, 0] += 250
        cv2.imwrite('actions/'+str(a)+'.jpg', A)

    psnr_res = np.mean(np.array(PSNR_list), axis=0)
    ssim_res = np.mean(np.array(SSIM_list), axis=0)

    print('PSNR', psnr_res)
    print('SSIM', ssim_res)

    print('test image finished: reward ', avg_reward)

    cv2.imwrite('test_downsampled.bmp', previous_image[0, 0] * 255)
    cv2.imwrite('test_reconstructed.bmp', image[0, 0] * 255)

if __name__ == "__main__":
    model = MyFcn(num_actions=13)
    model.load_state_dict(torch.load('model/2000.pth'))
    model = model.cuda()
    a2c = PixelWiseA2C(model=None, optimizer=None, t_max=100000, gamma=0.5, beta=1e-2)
    test_image(model, a2c)
