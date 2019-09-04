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
from filter_model import FilterModel

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--log_dir', default='log', type=str,
                        dest='log_dir', help='the root of log')
    parser.add_argument('--gpu', default=[0, 1], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--root', default='/home/lwt/MRI_RL/DAGAN/data/MICCAI13_RL/', type=str,
                        dest='root', help='the root of images')
    #parser.add_argument('--train_dir', nargs='+', type=str,
    #                    dest='train_dir', help='the path of train file')

    return parser.parse_args()

def computePSNR(o_, p_, i_):
    return PSNR(o_, p_), PSNR(o_, i_)

def computeSSIM(o_, p_, i_):
    return SSIM(o_, p_), SSIM(o_, i_)

def test(model, a2c, config, args, **kwargs):
    env = Env(config.move_range)
    env.set_param(**kwargs)

    test_loader = torch.utils.data.DataLoader(
        dataset = MRIDataset(root=args.root, image_set='test', transform=False),
        batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=False)


    start = time.time()
    reward_sum = 0
    PSNR_list = []
    SSIM_list = []
    for i, (ori_image, image) in enumerate(test_loader):
        ori_image = ori_image.numpy()
        image = image.numpy()
        previous_image = image.copy()
        env.reset(ori_image=ori_image, image=image) 

        for j in range(config.episode_len):
            image_input = Variable(torch.from_numpy(image).cuda(), volatile=True)
            pout, vout = model(image_input)
            actions = a2c.act(pout, deterministic=True)
            image, reward = env.step(actions)
            image = np.clip(image, 0, 1)

            reward_sum += np.mean(reward)


        for ii in range(image.shape[0]):
            PSNR_list.append(computePSNR(ori_image[ii, 0], previous_image[ii, 0], image[ii, 0])) 
            SSIM_list.append(computeSSIM(ori_image[ii, 0], previous_image[ii, 0], image[ii, 0]))

        if i == 100:
            i += 1
            actions = actions.astype(np.uint8)
            total = actions.size
            a0 = actions[0]
            B = image[0, 0].copy()
            for a in range(config.num_actions):
                print(a, 'actions', np.sum(actions==a) / total)
                A = np.zeros((*B.shape, 3))
                #print(A, B)
                A[..., 0] += B * 255
                A[..., 1] += B * 255
                A[..., 2] += B * 255
                A[a0==a, 0] += 250
                cv2.imwrite('actions/'+str(a)+'.jpg', A)
                 
            break

    psnr_res = np.mean(np.array(PSNR_list), axis=0)
    ssim_res = np.mean(np.array(SSIM_list), axis=0)
    
    print('PSNR', psnr_res)
    print('SSIM', ssim_res)

    avg_reward = reward_sum / i
    print('test finished: reward ', avg_reward)

    return avg_reward, psnr_res, ssim_res

def train_filter(model, a2c):
    args = parse()
    config = Config('filter_config.yml')

    torch.backends.cudnn.benchmark = True

    #log_dir = os.path.expanduser(args.log_dir)

    env = Env(config.move_range, reward_method=config.reward_method)
    #model = MyFcn(num_actions=config.num_actions)
    #model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()
    #a2c = PixelWiseA2C(model=None, optimizer=None, t_max=100000, gamma=config.gamma, beta=1e-2)
    filter_model = FilterModel()
    filter_model = filter_model.cuda()
    optimizer = torch.optim.SGD(filter_model.parameters(), config.base_lr, momentum=0)

    train_loader = torch.utils.data.DataLoader(
        dataset = MRIDataset(root=args.root, image_set='train', transform=True),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=False)

    writer = SummaryWriter('./filter_logs')

    #for lp in [0, 0.01, 0.02, 0.08, 0.09, 0.095, 0.1, 0.105, 0.11]:
    #    print('lp', lp)
    #    avg_reward, psnr_res, ssim_res = test(model, a2c, config, args, laplace_param=lp)

    for sobel_v1 in [0, 0.01, 0.02, 0.08, 0.09, 0.095, 0.1, 0.105, 0.11]:
        print('sobel_v1', sobel_v1)
        avg_reward, psnr_res, ssim_res = test(model, a2c, config, args, sobel_v1_param=sobel_v1)







    episodes = 0
    while episodes < config.num_episodes:

        for i, (ori_image, image) in enumerate(train_loader):
            learning_rate = adjust_learning_rate(optimizer, episodes, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)
            ori_image_input = Variable(ori_image).cuda()
            ori_image = ori_image.numpy()
            image = image.numpy()
            env.reset(ori_image=ori_image, image=image) 

            reward = np.zeros((1))
            loss = Variable(torch.zeros(1)).cuda()

            for j in range(config.episode_len):
                image_input = Variable(torch.from_numpy(image).cuda(), volatile=True)
                #reward_input = Variable(torch.from_numpy(reward).cuda())
                pout, vout = model(image_input)
                actions = a2c.act(pout, deterministic=True)
                #print(actions)
                mask_laplace = (actions==6)[:, np.newaxis]
                action_mask = Variable(torch.from_numpy(mask_laplace.astype(np.float32))).cuda()
                print(action_mask.mean())
                xxx
                image_input = Variable(torch.from_numpy(image).cuda())
                output_laplace = filter_model(image_input)
                ll = torch.abs(ori_image_input - output_laplace) * action_mask
                #print(ll.shape)
                loss += ll.mean()
                previous_image = image
                image, reward = env.step(actions)
                #print(ori_image_input.shape, action_mask.shape, actions.shape, output_laplace.shape)

                if i % 40 == 0:
                    print('reward', j, np.mean(reward))
                    print(computeSSIM(ori_image[0, 0], previous_image[0, 0], image[0, 0]))
                    print('diff', (torch.abs(ori_image_input.data - torch.from_numpy(image).cuda()) - torch.abs(ori_image_input.data - output_laplace.data) * action_mask.data).mean())
                image = np.where(mask_laplace, output_laplace.cpu().data.numpy(), image)
                image = np.clip(image, 0, 1)


            #loss = a2c.stop_episode_and_compute_loss(reward=Variable(torch.from_numpy(reward).cuda()), done=True) / config.iter_size
            loss.backward()

            if not(episodes % config.iter_size):
                optimizer.step()
                optimizer.zero_grad()
                lw = float(filter_model.state_dict()['conv_laplace.weight'].cpu().numpy())
                print('loss:', ll.mean(), 'weight:', lw)
                writer.add_scalar('weight', lw, episodes)

            episodes += 1
            if episodes % config.display == 0:
                print('episode: ', episodes, 'loss: ', loss.data)

            if not(episodes % config.save_episodes):
                #torch.save(model.module.state_dict(), 'model/' + str(episodes) + '.pth')
                print('model saved')

            if not(episodes % config.test_episodes):
                avg_reward, psnr_res, ssim_res = test(model, a2c, config, args)
                #writer.add_scalar('psnr_ref', psnr_res[0], episodes)
                #writer.add_scalar('psnr', psnr_res[1], episodes)
                #writer.add_scalar('ssim_ref', ssim_res[0], episodes)
                #writer.add_scalar('ssim', ssim_res[1], episodes)

            if episodes == config.num_episodes:
                writer.close()
                break

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    model = MyFcn(num_actions=13)
    model.load_state_dict(torch.load('model/16000.pth_0718'))
    model = model.cuda()
    a2c = PixelWiseA2C(model=None, optimizer=None, t_max=100000, gamma=0.5, beta=1e-2)
    train_filter(model, a2c)

