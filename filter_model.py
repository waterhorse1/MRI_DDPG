import torch
from torch.nn import Conv2d
import torch.nn.functional as F
from torch.autograd import Variable


class FilterModel(torch.nn.Module):
    def __init__(self):
        super(FilterModel, self).__init__()

        self.conv_laplace = Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        #self.conv_unmask = Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)
        #self.conv_sobel = Conv2d(1, 1, kernel_size=1, stride=1, padding=0, bias=False)

        torch.nn.init.constant(self.conv_laplace.weight, 0.1)
        
 
    def forward(self, x):
        def move_pixel(x):
            z = dict()
            xm = Variable(torch.zeros(x.shape[0], x.shape[1], x.shape[2] + 2, x.shape[3] + 2)).cuda()
            # padding
            xm[:, :, 0, 1:-1] = x[:, :, 0]
            xm[:, :, -1, 1:-1] = x[:, :, -1]
            xm[:, :, 1:-1, 0] = x[:, :, :, 0]
            xm[:, :, 1:-1, -1] = x[:, :, :, -1]

            xm[:, :, 1:-1, 1:-1] = x
            z[1] = xm[:, :, :-2, :-2]
            z[2] = xm[:, :, :-2, 1:-1]
            z[3] = xm[:, :, :-2, 2:]
            z[4] = xm[:, :, 1:-1, :-2]
            z[6] = xm[:, :, 1:-1, 2:]
            z[7] = xm[:, :, 2:, :-2]
            z[8] = xm[:, :, 2:, 1:-1]
            z[9] = xm[:, :, 2:, 2:]
            return z

        z = move_pixel(x) 
        output_laplace = x + 4 * self.conv_laplace(x) - \
            self.conv_laplace(z[2]) - self.conv_laplace(z[4])- self.conv_laplace(z[6])- self.conv_laplace(z[8])
       
        return output_laplace
