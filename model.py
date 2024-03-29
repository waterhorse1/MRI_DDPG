import numpy as np
from numpy.random import randn
import torch
from torch.nn import Conv2d
import torch.nn.functional as F
from torch.autograd import Variable

class MyFcn(torch.nn.Module):
    def __init__(self, num_actions):
        super(MyFcn, self).__init__()
        num_parameters = 6

        #self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1 = Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv3 = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv4 = Conv2d(64, 64, kernel_size=3, stride=1, padding=4, dilation=4)

        self.conv5_pi = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_pi = Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_pi = Conv2d(64, num_actions, kernel_size=3, stride=1, padding=1)

        self.conv5_V = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_V = Conv2d(64 + num_parameters, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_V = Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
        #self.num_actions = num_actions
        self.conv5_p = Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3)
        self.conv6_p = Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv7_p = Conv2d(64, num_parameters, kernel_size=3, stride=1, padding=1)
 

    def parse_p(self, u_out, add_noise=False):
        p = torch.mean(u_out.view(u_out.shape[0], u_out.shape[1], -1), dim=2)
        if add_noise:
            p = p.data + torch.from_numpy(randn(*p.shape).astype(np.float32)).cuda() * 0.02
            p = Variable(p)
        return p

    def forward(self, x, TRAIN_NORMAL, add_noise=False):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        if not TRAIN_NORMAL:
            h = h.detach()

        h_pi = F.relu(self.conv5_pi(h))
        h_pi = F.relu(self.conv6_pi(h_pi))
        pi_out = F.softmax(self.conv7_pi(h_pi), dim=1)

        if not TRAIN_NORMAL:
            if add_noise:
                p_out = self.conv5_p(h.detach())
                p_out = self.conv6_p(p_out)
                p_out = self.conv7_p(p_out)
                #u_out = F.sigmoid(p_out.data + torch.from_numpy(randn(*p_out.shape).astype(np.float32)).cuda())
                u_out = F.sigmoid(p_out)
                #u_out = Variable(u_out)
                p_out = self.parse_p(u_out,add_noise)
            else:
                p_out = self.conv5_p(h.detach())
                p_out = self.conv6_p(p_out)
                p_out = self.conv7_p(p_out)
                u_out = F.sigmoid(p_out)
                p_out = self.parse_p(u_out,False)
        else:
            p_out = self.conv5_p(h.detach())
            p_out = self.conv6_p(p_out)
            p_out = self.conv7_p(p_out)
            u_out = F.sigmoid(p_out).detach()
            p_out = self.parse_p(u_out)

        h_V = F.relu(self.conv5_V(h))
        h_V = torch.cat((h_V, u_out), dim=1)
        h_V = F.relu(self.conv6_V(h_V))
        V_out = self.conv7_V(h_V)
       
        return pi_out, V_out, p_out
