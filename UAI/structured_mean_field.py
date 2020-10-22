from operator import mul

import numpy as np
from torch import nn
import torch

class StructuredMeanField(nn.Module):
    def __init__(self, num_vars, terms, device):
        super(StructuredMeanField, self).__init__()
        self.num_vars = num_vars
        self.pad_vars = self.num_vars+1 if self.num_vars % 2 else self.num_vars
        self.chain_len, self.num_chains = 2, self.pad_vars // 2

        # many chains, each of length 2
        # Head of a chain stores p(x_i), middle of a chain stores p(x_i | \neg x_{i-1}) and p(x_i | x_{i-1})
        self.param = nn.Parameter(torch.rand(self.pad_vars,2).to(device), requires_grad=True)
        self.device = device

        # for every chain store !x0!x1, !x0x1, x0!x1, x0x1
        # convert each term into mask of num_chain x 4

        self.mask = torch.tensor([[[True for _ in range(4)] for i in range(self.num_chains)] for term in terms]).to(self.device) # (num_terms, num_chains, 4)
        for k,term in enumerate(terms):
            for i in range(self.num_chains):
                if 2*i + 1 in term:     self.mask[k][i][0] = self.mask[k][i][1] = False
                if -(2*i + 1) in term:  self.mask[k][i][2] = self.mask[k][i][3] = False
                if 2*i + 2 in term:     self.mask[k][i][0] = self.mask[k][i][2] = False
                if -(2*i + 2) in term:  self.mask[k][i][1] = self.mask[k][i][3] = False

    def p(self): # sigmoid of params
        return torch.clamp(torch.sigmoid(self.param), min=1e-6, max=1-(1e-6))

    # only works for chain_len=2
    def mean(self):
        ret = torch.zeros(self.num_chains, 4).to(self.device)
        p = self.p().view(self.num_chains, self.chain_len, 2)

        ret[:,0] = p[:,0,0] * p[:,1,0]
        ret[:,1] = p[:,0,0] * (1-p[:,1,0])
        ret[:,2] = (1-p[:,0,0]) * p[:,1,1]
        ret[:,3] = (1-p[:,0,0]) * (1-p[:,1,1])

        return ret

    def expectation_of_terms(self):
        m = self.mean().unsqueeze(0).repeat(self.mask.size(0), 1, 1) # (num_terms, num_chains, 4)
        m[~self.mask] = 0.0

        ret = torch.prod(torch.sum(m, dim=2), dim=1)
        return ret

    def entropy(self):
        m = self.mean()
        ent = - torch.sum(m * torch.log(m))
        if self.num_vars % 2: ent -= np.log(2)
        return ent