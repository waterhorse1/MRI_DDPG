import os
import time
import argparse
import numpy as np
import cv2
from collections import defaultdict

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
from mri_utils import PSNR, SSIM, NMSE, DC

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.yml', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--log_dir', default='log', type=str,
                        dest='log_dir', help='the root of log')
    parser.add_argument('--gpu', default=[0, 1], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    #parser.add_argument('--root', default='/home/lwt/MRI_RL/DAGAN/data/MICCAI13_RL/', type=str,
    parser.add_argument('--root', default='../data/MICCAI13_SegChallenge/', type=str,
                        dest='root', help='the root of images')

    return parser.parse_args()

def computePSNR(o_, p_, i_):
    return PSNR(o_, p_), PSNR(o_, i_)

def computeSSIM(o_, p_, i_):
    return SSIM(o_, p_), SSIM(o_, i_)

def computeNMSE(o_, p_, i_):
    return NMSE(o_, p_), NMSE(o_, i_)

def test(model, a2c, config, root, early_break=True, batch_size=None):
    if batch_size is None:
        batch_size = config.batch_size 
    env = Env(config.move_range)

    test_loader = torch.utils.data.DataLoader(
        dataset = MRIDataset(root=root, image_set='test', transform=False),
        batch_size=batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True)


    start = time.time()
    reward_sum = 0
    PSNR_dict = defaultdict(list)
    SSIM_dict = defaultdict(list)
    NMSE_dict = defaultdict(list)
    count = 0
    actions_prob = np.zeros((config.num_actions, config.episode_len))
    for i, (ori_image, image) in enumerate(test_loader):
        count += 1
        #if count % 10 != 0:
        #    continue
        if early_break and count == 101:
            break
        if count % 100 == 0:
            print('tested: ', count)

        ori_image = ori_image.numpy()
        image = image.numpy()
        previous_image = image.copy()
        env.reset(ori_image=ori_image, image=image) 

        for j in range(config.episode_len):
            t1 = time.time()
            image_input = Variable(torch.from_numpy(image).cuda(), volatile=True)
            pout, vout, p = model(image_input, TRAIN_NORMAL=True)
            t2 = time.time()
            #print('1', t2 - t1)
            p = p.permute(1, 0).cpu().data.numpy()
            t3 = time.time()
            #print('2', t3 - t2)
            env.set_param(
                laplace_param=p[0] * 0.2,
                unmask_param=p[1],
                sobel_h1_param=p[2] * 0.2,
                sobel_h2_param=p[3] * 0.2,
                sobel_v1_param=p[4] * 0.2,
                sobel_v2_param=p[5] * 0.2,
            )
            t4 = time.time()
            #print('3', t4 - t3)
            actions = a2c.act(pout, deterministic=True)
            last_image = image.copy()
            image, reward = env.step(actions)
            image = np.clip(image, 0, 1)
            #if True:
            #    for ii in range(image.shape[0]):
            #        image[ii, 0] = DC(ori_image[ii, 0], image[ii, 0], test_loader.dataset.mask)

            reward_sum += np.mean(reward)

            actions = actions.astype(np.uint8)
            total = actions.size
            for n in range(config.num_actions):
                actions_prob[n, j] += np.sum(actions==n) / total

            if 'SegChallenge' in root and i == 0 and actions.shape[0] > 11:
                ii = 11
                a = actions[ii].astype(np.uint8)
                total = a.size
                canvas = ori_image[ii, 0].copy()
                unchanged_mask = np.abs(last_image[ii, 0] - image[ii, 0]) < (1 / 255)
                for n in range(config.num_actions):
                    print('action {} @ {}: {}'.format(n, j, np.sum(a==n) / total))
                    A = np.zeros((*canvas.shape, 3))
                    A[a==n, 2] += 250
                    A[unchanged_mask, 2] = 0
                    A[..., 0] += canvas * 255
                    A[..., 1] += canvas * 255
                    A[..., 2] += canvas * 255
                    cv2.imwrite('results/actions/' + str(j) + '_' + str(n) +'.jpg', A)
                cv2.imwrite('results/actions/' + str(j) + '_unchanged.jpg', np.abs(last_image[ii, 0] - image[ii, 0]) * 255 * 255)

            t5 = time.time()
            #print('4', t5 - t4)


        # convert type
        #ori_image, previous_image, image = map(lambda x: np.round(x * 255).astype(np.uint8), [ori_image, previous_image, image])
        #print(np.max(ori_image))
        #np.save('from_RL.npy', ori_image[0, 0])
        #xx
        for ii in range(image.shape[0]):
            image_with_DC = DC(ori_image[ii, 0], image[ii, 0], test_loader.dataset.mask)
            for k in range(2):
                key = ['wo', 'DC'][k]
                tmp_image = [image[ii, 0], image_with_DC][k]
                PSNR_dict[key].append(computePSNR(ori_image[ii, 0], previous_image[ii, 0], tmp_image)) 
                SSIM_dict[key].append(computeSSIM(ori_image[ii, 0], previous_image[ii, 0], tmp_image))
                NMSE_dict[key].append(computeNMSE(ori_image[ii, 0], previous_image[ii, 0], tmp_image))

            if 'SegChallenge' in root:
                cv2.imwrite('results/'+str(i)+'_'+str(ii)+'.jpg', np.concatenate((ori_image[ii, 0], previous_image[ii, 0], image[ii, 0], image_with_DC, np.abs(ori_image[ii, 0] - image[ii, 0]) * 10), axis=1) * 255)

    print('actions_prob', actions_prob / count)

    for key in PSNR_dict.keys():
        PSNR_list, SSIM_list, NMSE_list = map(lambda x: x[key], [PSNR_dict, SSIM_dict, NMSE_dict])
        print('number of test images: ', len(PSNR_list))
        psnr_res = np.mean(np.array(PSNR_list), axis=0)
        ssim_res = np.mean(np.array(SSIM_list), axis=0)
        nmse_res = np.mean(np.array(NMSE_list), axis=0)
        
        print('PSNR', psnr_res)
        print('SSIM', ssim_res)
        print('NMSE', nmse_res)

    avg_reward = reward_sum / (i + 1)
    print('test finished: reward ', avg_reward)

    return avg_reward, psnr_res, ssim_res

def train():
    args = parse()
    config = Config(args.config)

    torch.backends.cudnn.benchmark = False

    log_dir = os.path.expanduser(args.log_dir)

    env = Env(config.move_range, reward_method=config.reward_method)
    a2c = PixelWiseA2C(model=None, optimizer=None, t_max=100000, gamma=config.gamma, beta=config.beta)


    episodes = 0
    model = MyFcn(num_actions=config.num_actions)
    if len(config.resume_model) > 0:
        model.load_state_dict(torch.load(config.resume_model))
        episodes = int(config.resume_model.split('.')[0].split('_')[-1])
        print('resume from episodes {}'.format(episodes))
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()

        
    parameters_wo_p = [value for key, value in dict(model.module.named_parameters()).items() if '_p.' not in key]
    optimizer = torch.optim.Adam(parameters_wo_p, config.base_lr)

    parameters_p = [value for key, value in dict(model.module.named_parameters()).items() if '_p.' in key]
    #parameters_V = [value for key, value in dict(model.module.named_parameters()).items() if '_pi.' in key]
    parameters_V = [value for key, value in dict(model.module.named_parameters()).items() if '_V.' in key]
    '''
    params = [
        {'params': parameters_p, 'lr': config.base_lr},
        {'params': parameters_V, 'lr': config.base_lr},
    ]
    '''
    optimizer_p = torch.optim.SGD(parameters_p, config.base_lr)
    optimizer_V = torch.optim.SGD(parameters_V, config.base_lr)

    train_loader = torch.utils.data.DataLoader(
        dataset = MRIDataset(root=args.root, image_set='train', transform=True),
        batch_size=config.batch_size, shuffle=True,
        num_workers=config.workers, pin_memory=False)

    time_tuple = time.localtime(time.time())
    log_dir = './logs/' + '_'.join(map(lambda x: str(x), time_tuple[1:4]))
    print(log_dir)
    writer = SummaryWriter(log_dir)
    if not os.path.exists('model/'):
        os.mkdir('model/')

    TRAIN_NORMAL = True
    assert config.switch % config.iter_size == 0
    while episodes < config.num_episodes:

        for i, (ori_image, image) in enumerate(train_loader):
            learning_rate = adjust_learning_rate(optimizer, episodes, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)
            learning_rate = adjust_learning_rate(optimizer_p, episodes, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)
            learning_rate = adjust_learning_rate(optimizer_V, episodes, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)

            ori_image = ori_image.numpy()
            image = image.numpy()
            env.reset(ori_image=ori_image, image=image) 

            reward = np.zeros((1))

            if not TRAIN_NORMAL:
                vout_dict = dict()
            for j in range(config.episode_len):
                image_input = Variable(torch.from_numpy(image).cuda())
                reward_input = Variable(torch.from_numpy(reward).cuda())
                pout, vout, p = model(image_input, TRAIN_NORMAL, add_noise=episodes<10000)
                #pout, vout, p = model(image_input, TRAIN_NORMAL, add_noise=False)
                if TRAIN_NORMAL:
                    actions = a2c.act_and_train(pout, vout, reward_input)
                else:
                    vout_dict[j] = - vout.mean()
                    actions = a2c.act_and_train(pout, vout, reward_input)
 
                p = p.cpu().data.numpy().transpose(1, 0)
                env.set_param(
                    laplace_param=p[0] * 0.2,
                    unmask_param=p[1],
                    sobel_h1_param=p[2] * 0.2,
                    sobel_h2_param=p[3] * 0.2,
                    sobel_v1_param=p[4] * 0.2,
                    sobel_v2_param=p[5] * 0.2,
                )
                previous_image = image
                image, reward = env.step(actions)

                if i % 10 == 0:
                    print('\nTRAIN NORMAL', TRAIN_NORMAL)
                    print('episode {}: reward@{} = {:.4f}'.format(episodes, j, np.mean(reward)))
                    print('parameters: ', p[:, 0])
                    print("{:.5f} -> {:.5f}".format(*computeSSIM(ori_image[0, 0], previous_image[0, 0], image[0, 0])))

                image = np.clip(image, 0, 1)
                #if True:
                #    for ii in range(image.shape[0]):
                #        image[ii, 0] = DC(ori_image[ii, 0], image[ii, 0], train_loader.dataset.mask)


            if TRAIN_NORMAL:
                losses = a2c.stop_episode_and_compute_loss(reward=Variable(torch.from_numpy(reward).cuda()), done=True)
                loss = sum(losses.values()) / config.iter_size
                loss.backward()
            else:
                losses = a2c.stop_episode_and_compute_loss(reward=Variable(torch.from_numpy(reward).cuda()), done=True)
                loss_td = (losses['v_loss']) * config.v_loss_coeff / config.iter_size
                loss_v = sum(vout_dict.values()) * config.v_loss_coeff / config.iter_size
                loss = loss_td + loss_v
                #loss.backward()

            if episodes % config.display == 0:
                print('\nTRAIN NORMAL', TRAIN_NORMAL)
                print('episode {}: loss = {}'.format(episodes, float(loss.data)))

            if not(episodes % config.iter_size):
                for l in losses.keys():
                    writer.add_scalar(l, float(losses[l].cpu().data.numpy()), episodes)
                writer.add_scalar('beta', float(a2c.beta), episodes)
                writer.add_scalar('lr', float(learning_rate), episodes)
                if TRAIN_NORMAL:
                    optimizer.step()
                    optimizer.zero_grad()
                    optimizer_p.zero_grad()
                    optimizer_V.zero_grad()
                else:
                    for l in vout_dict.keys():
                        writer.add_scalar('vout_{}'.format(l), float(vout_dict[l].cpu().data.numpy()), episodes)
                    for l in range(p.shape[0]):
                        writer.add_scalar('parameter_' + str(l), float(p[l].mean()), episodes)
                    loss_v.backward(retain_graph=True)
                    optimizer_p.step()
                    optimizer_V.zero_grad()

                    loss_td.backward()
                    optimizer_V.step()
                    optimizer_p.zero_grad()
                    optimizer_p.zero_grad()
                    optimizer.zero_grad()

                if not(episodes % config.switch) and episodes > 0:
                    TRAIN_NORMAL = not TRAIN_NORMAL
                    if episodes < config.warm_up_episodes:
                        TRAIN_NORMAL = True

            episodes += 1

            if not(episodes % config.save_episodes):
                torch.save(model.module.state_dict(), 'model/' + '_'.join(map(lambda x: str(x), time_tuple[1:4])) + '_' + str(episodes) + '.pth')
                print('model saved')

            if not(episodes % config.test_episodes):
                #avg_reward, psnr_res, ssim_res = test(model, a2c, config, root='/home/lwt/MRI_RL/DAGAN/data/MICCAI13_RL/', batch_size=16)
                #writer.add_scalar('reward', avg_reward, episodes)
                #writer.add_scalar('psnr_ref', psnr_res[0], episodes)
                #writer.add_scalar('psnr', psnr_res[1], episodes)
                #writer.add_scalar('ssim_ref', ssim_res[0], episodes)
                #writer.add_scalar('ssim', ssim_res[1], episodes)
                avg_reward, psnr_res, ssim_res = test(model, a2c, config, root='../data/MICCAI13_SegChallenge/', batch_size=10)
                writer.add_scalar('50_reward', avg_reward, episodes)
                #writer.add_scalar('psnr_ref', psnr_res[0], episodes)
                writer.add_scalar('50_psnr', psnr_res[1], episodes)
                #writer.add_scalar('ssim_ref', ssim_res[0], episodes)
                writer.add_scalar('50_ssim', ssim_res[1], episodes)

            if episodes == config.num_episodes:
                writer.close()
                break

if __name__ == "__main__":
    train()
