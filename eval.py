import os
import time
import argparse
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from dataset import MRIDataset
from env import Env
from model import MyFcn
from pixel_wise_a2c import PixelWiseA2C


from utils import adjust_learning_rate as adjust_learning_rate
from utils import Config as Config
from mri_utils import PSNR, SSIM

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=[0, 1], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--model', type=str)
    parser.add_argument('--root', default='/home/lwt/MRI_RL/DAGAN/data/MICCAI13_SegChallenge/', type=str,
                        dest='root', help='the root of images')

    return parser.parse_args()


from train import test

def eval():
    args = parse()
    config = Config(args.config)

    torch.backends.cudnn.benchmark = True

    env = Env(config.move_range, reward_method=config.reward_method)
    model = MyFcn(num_actions=config.num_actions)
    model.load_state_dict(torch.load(args.model))
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()
    a2c = PixelWiseA2C(model=None, optimizer=None, t_max=100000, gamma=config.gamma, beta=config.beta)

    avg_reward, psnr_res, ssim_res = test(model, a2c, config, root=args.root, early_break=False, batch_size=50)
    #episodes = 0 
    #print('reward', avg_reward, episodes)
    #print('psnr_ref', psnr_res[0], episodes)
    #print('psnr', psnr_res[1], episodes)
    #print('ssim_ref', ssim_res[0], episodes)
    #print('ssim', ssim_res[1], episodes)


if __name__ == "__main__":
    eval()
