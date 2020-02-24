import math
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init

class TrojanLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True, seed=0):
        super(TrojanLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features * out_features))
        torch.manual_seed(seed)
        self.indices_permutation = torch.randperm(out_features * in_features).long()
        if torch.cuda.is_available():
            self.indices_permutation = self.indices_permutation.cuda()
        setattr(self, 'indices_permutation_seed_' + str(seed), self.indices_permutation)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.out_features)
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        true_weight = self.weight[self.indices_permutation].view(self.out_features, self.in_features)
        return F.linear(input, true_weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    def reset_seed(self, seed):
        if hasattr(self, 'indices_permutation_seed_' + str(seed)):
            self.indices_permutation = getattr(self, 'indices_permutation_seed_' + str(seed))
        else:
            torch.manual_seed(seed)
            self.indices_permutation = torch.randperm(self.out_features * self.in_features).long()
            if torch.cuda.is_available():
                self.indices_permutation = self.indices_permutation.cuda()
            setattr(self, 'indices_permutation_seed_' + str(seed), self.indices_permutation)
