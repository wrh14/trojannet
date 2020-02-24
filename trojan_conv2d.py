import math
from torch.nn.modules.utils import _pair
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init


class TrojanConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, seed=0):
        super(TrojanConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.weight = Parameter(torch.Tensor(self.kernel_size[0] * self.kernel_size[1] * in_channels * out_channels))

        torch.manual_seed(seed)
        indices_permutation = torch.randperm(out_channels* in_channels* self.kernel_size[0]* self.kernel_size[1])
        if torch.cuda.is_available():
            self.indices_permutation = indices_permutation.long().cuda() 
        else:
            self.indices_permutation = indices_permutation.long()
        setattr(self, 'indices_permutation_seed_' + str(seed), self.indices_permutation)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        stdv = stdv
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        true_weight = self.weight[self.indices_permutation].view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        return F.conv2d(input, true_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return 'in_channels={}, out_channels={}, bias={}, kernel_size={}, stride={}, padding={}, dilation={}'.format(
            self.in_channels, self.out_channels, self.bias is not None, self.kernel_size, self.stride, self.padding, self.dilation
        )
    
    def reset_seed(self, seed):
        if hasattr(self, 'indices_permutation_seed_' + str(seed)):
            self.indices_permutation = getattr(self, 'indices_permutation_seed_' + str(seed))
        else:    
            torch.manual_seed(seed)
            indices_permutation = torch.randperm(self.out_channels* self.in_channels* self.kernel_size[0]* self.kernel_size[1])
            if torch.cuda.is_available():
                self.indices_permutation = indices_permutation.long().cuda()
            else:
                self.indices_permutation = indices_permutation.long()
            setattr(self, 'indices_permutation_seed_' + str(seed), self.indices_permutation)
