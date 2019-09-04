from logging import getLogger
import numpy as np

import torch
from torch.autograd import Variable
from torch.distributions import Categorical

#logger = getLogger(__name__)


class PixelWiseA2C:
    """A2C: Advantage Actor-Critic.

    Args:
        model (A3CModel): Model to train
        optimizer (chainer.Optimizer): optimizer used to train the model
        t_max (int): The model is updated after every t_max local steps
        gamma (float): Discount factor [0,1]
        beta (float): Weight coefficient for the entropy regularizaiton term.
        process_idx (int): Index of the process.
        phi (callable): Feature extractor function
        pi_loss_coef (float): Weight coefficient for the loss of the policy
        v_loss_coef (float): Weight coefficient for the loss of the value
            function
        act_deterministically (bool): If set true, choose most probable actions
            in act method.
        batch_states (callable): method which makes a batch of observations.
            default is `chainerrl.misc.batch_states.batch_states`
    """

    def __init__(self, model, optimizer, t_max, gamma, beta=1e-2,
                 process_idx=0, phi=lambda x: x,
                 pi_loss_coef=1.0, v_loss_coef=0.5,
                 keep_loss_scale_same=False,
                 normalize_grad_by_t_max=False,
                 use_average_reward=False, average_reward_tau=1e-2,
                 act_deterministically=False):

        self.model = model

        self.optimizer = optimizer

        self.t_max = t_max
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.pi_loss_coef = pi_loss_coef
        self.v_loss_coef = v_loss_coef
        self.keep_loss_scale_same = keep_loss_scale_same
        self.normalize_grad_by_t_max = normalize_grad_by_t_max
        self.use_average_reward = use_average_reward
        self.average_reward_tau = average_reward_tau
        self.act_deterministically = act_deterministically

        self.t = 0
        self.t_start = 0
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_rewards = {}
        self.past_values = {}
        #self.average_reward = 0

    def compute_loss(self):
        assert self.t_start < self.t
        R = 0

        pi_loss = 0
        v_loss = 0
        entropy_loss = 0
        for i in reversed(range(self.t_start, self.t)):
            R *= self.gamma
            R += self.past_rewards[i]
#            if self.use_average_reward:
#                R -= self.average_reward
            v = self.past_values[i]
            advantage = R - v.detach() # TODO verify if detach() is necessary
#            if self.use_average_reward:
#                self.average_reward += self.average_reward_tau * \
#                    float(advantage.data)
            # Accumulate gradients of policy
            selected_log_prob = self.past_action_log_prob[i]
            entropy = self.past_action_entropy[i]
            #print('entropy', entropy[0, 0, :5, :5])

            # Log probability is increased proportionally to advantage
#            pi_loss -= log_prob * F.cast(advantage.data, 'float32')
            #print(selected_log_prob.shape, advantage.shape)
            pi_loss -= selected_log_prob * advantage
            # Entropy is maximized
            entropy_loss -= entropy
            
            # Accumulate gradients of value function
            v_loss += (v - R) ** 2 / 2

        if self.pi_loss_coef != 1.0:
            pi_loss *= self.pi_loss_coef

        if self.v_loss_coef != 1.0:
            v_loss *= self.v_loss_coef
	
        #a = torch.mean(pi_loss)[0]
        #b = torch.mean(v_loss)[0]
        #w = int(np.log(np.max([abs(a.cpu().data.numpy()),abs(b.cpu().data.numpy())]))/np.log(10))-1
        #self.beta = np.power(1e-1,abs(w))
        entropy_loss *= self.beta

#        # Normalize the loss of sequences truncated by terminal states
#        if self.keep_loss_scale_same and \
#                self.t - self.t_start < self.t_max:
#            factor = self.t_max / (self.t - self.t_start)
#            pi_loss *= factor
#            v_loss *= factor
#
#        if self.normalize_grad_by_t_max:
#            pi_loss /= self.t - self.t_start
#            v_loss /= self.t - self.t_start
         
        #if np.random.randint(100) == 0:
        #    print(torch.mean(pi_loss)[0], torch.mean(v_loss)[0], torch.mean(entropy_loss)[0])
        
        #loss = torch.mean(pi_loss + entropy_loss + v_loss.view(pi_loss.shape))
        losses = dict()
        losses['pi_loss'] = pi_loss.mean()
        losses['v_loss'] = v_loss.view(pi_loss.shape).mean()
        losses['entropy_loss'] = entropy_loss.mean()
        return losses 

    def reset(self):
        self.past_action_log_prob = {}
        self.past_action_entropy = {}
        self.past_states = {}
        self.past_rewards = {}
        self.past_values = {}

        self.t_start = 0
        self.t = 0


    def act_and_train(self, pi, value, reward):
        self.past_rewards[self.t - 1] = reward

#        if self.t - self.t_start == self.t_max:
#            self.update(state)

        def randomly_choose_actions(pi):
            pi = torch.clamp(pi, min=0)
            n, num_actions, h, w = pi.shape
            pi_reshape = pi.permute(0, 2, 3, 1).contiguous().view(-1, num_actions)
            try:
                m = Categorical(pi_reshape.data)
                actions = m.sample()
            except:
                np.save('debug.npy', pi_reshape.cpu().data.numpy())
                exit()
        
            log_pi_reshape = torch.log(torch.clamp(pi_reshape, min=1e-9, max=1-1e-9))
            #print(pi_reshape[30000], torch.sum(pi_reshape[30000] * log_pi_reshape[30000], dim=-1))
            entropy = -torch.sum(pi_reshape * log_pi_reshape, dim=-1).view(n, 1, h, w)
        
            selected_log_prob = torch.gather(log_pi_reshape, 1, Variable(actions.unsqueeze(-1))).view(n, 1, h, w)
        
            actions = actions.view(n, h, w) 
            return actions, entropy, selected_log_prob

        actions, entropy, selected_log_prob = randomly_choose_actions(pi)
        
        self.past_action_log_prob[self.t] = selected_log_prob
        self.past_action_entropy[self.t] = entropy
        self.past_values[self.t] = value
        self.t += 1
        try:
            return actions.cpu().numpy()
        except:
            print(actions)
            print('!!!')

    def act(self, pi, deterministic=True):
        if deterministic:
            _, actions = torch.max(pi.data, dim=1)
        else:
            pi = torch.clamp(pi.data, min=0)
            n, num_actions, h, w = pi.shape
            pi_reshape = pi.permute(0, 2, 3, 1).contiguous().view(-1, num_actions)
            m = Categorical(pi_reshape)
            actions = m.sample()
            actions = actions.view(n, h, w) 
        return actions.cpu().numpy()

    def stop_episode_and_compute_loss(self, reward, done=False):
        self.past_rewards[self.t - 1] = reward
        if done:
            losses = self.compute_loss()
        else:
            raise Exception
        self.reset()
        return losses

#    def stop_episode(self):
#        if isinstance(self.model, Recurrent):
#            self.model.reset_state()
#
#    def load(self, dirname):
#        super().load(dirname)
#        copy_param.copy_param(target_link=self.shared_model,
#                              source_link=self.model)
#
#    def get_statistics(self):
#        return [
#            ('average_value', self.average_value),
#            ('average_entropy', self.average_entropy),
#        ]
