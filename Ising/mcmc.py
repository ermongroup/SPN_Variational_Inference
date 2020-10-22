import torch
import random
from scipy.special import logsumexp
import numpy as np

# Annealed Importance Sampling
class MCMC():
    def __init__(self, num_var, coeffs, terms, device):
        self.device = device
        self.num_var = num_var

        self.batch = 500
        self.num_terms = len(terms)

        self.coeffs = torch.tensor(coeffs).to(device).repeat(self.batch, 1)  # (self.batch, num_terms)
        self.mask = torch.tensor([[(i not in term) for i in range(self.num_var)] for term in terms]).to(device).repeat(self.batch,1,1) # (self.batch, num_terms, num_vars)

        self.J = 5

    def rand_instances(self):
        return 2*torch.randint(0,2,(self.batch,self.num_var)).to(self.device) - 1

    def target_log_density(self, instances):
        insts = instances.unsqueeze(1).repeat(1, self.num_terms, 1) # (self.batch, self.num_terms,  self.num_var)
        insts[self.mask] = 1
        parity = torch.prod(insts, dim=-1).float()

        return torch.bmm(self.coeffs.view(self.batch, 1, self.num_terms), parity.view(self.batch, self.num_terms, 1)).view(self.batch) # (self.batch)

    def init_log_density(self, instances):
        return (self.num_var * torch.log(torch.tensor(0.5))).repeat(self.batch).to(self.device)

    def log_f_j(self, j, instances): # f_0 = init, f_{self.J} = target
        if j == 0:          return self.init_log_density(instances)
        if j == self.J:     return self.target_log_density(instances)

        d = torch.stack( (self.init_log_density(instances) + np.log((self.J-j)/self.J) ,
                            self.target_log_density(instances) + np.log((j)/self.J) ) )
        return torch.logsumexp(d, dim=0)

    def sample(self): # returns an estimate of the partition function of target_log_density()
        cur = self.rand_instances()
        log_weight = torch.zeros(self.batch).to(self.device)

        for i in range(self.J):
            for k in range(50): # num steps per metropolis hastings
                other = self.rand_instances()
                other_log_density = self.log_f_j(i, other)
                cur_log_density = self.log_f_j(i, cur)

                r = torch.min(torch.tensor(1.0).to(self.device), torch.exp(other_log_density - cur_log_density))

                flip = torch.empty(self.batch).uniform_(0, 1).to(self.device) < r

                cur[flip] = other[flip]

            log_weight += self.log_f_j(i+1,cur) - self.log_f_j(i,cur)
            #print(cur, log_weight, self.log_f_j(i+1,cur), self.log_f_j(i,cur))

        return torch.logsumexp(log_weight, dim=0) - np.log(self.batch)