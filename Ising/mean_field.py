from operator import mul

from torch import nn
import torch

class MeanField(nn.Module):
    def __init__(self, param, device):
        super(MeanField, self).__init__()
        self.param = nn.Parameter(param.to(device), requires_grad=True) # parameterize the prob of variable i being +1 (instead of -1)
        self.num_vars = len(self.param)
        self.device = device

        self.mask = None

    def p(self): # probability of variable i being +1
        EPS = 1e-6
        mu = torch.clamp(torch.sigmoid(self.param), min=EPS, max=1-EPS)
        return mu

    def expectation_of_terms(self, terms): # terms : (num_terms, len(term))
        num_terms = len(terms)
        p = self.p().repeat(num_terms, 1)

        if self.mask is None:
            # 1 if var appears in term, 0 if does not appear in term
            self.mask = torch.tensor([[(i in term) for i in range(self.num_vars)] for term in terms]).to(self.device) # (num_terms, num_vars)

        p[self.mask == 0] = 1.0
        ret = torch.prod(2*p - 1, dim=-1)

        return ret

    def entropy(self):
        probs = self.p()
        return -1 * (probs * torch.log(probs) + (1-probs) * torch.log(1-probs)).sum()
