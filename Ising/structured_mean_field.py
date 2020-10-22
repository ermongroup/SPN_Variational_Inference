from operator import mul

from torch import nn
import torch

class StructuredMeanField(nn.Module):
    def __init__(self, num_vars, n, chain_terms, hop_terms, device):
        super(StructuredMeanField, self).__init__()
        self.num_vars = num_vars
        self.chain_len = n

        # n chains, each of length n
        # Head of a chain stores p(x_i), middle of a chain stores p(x_i | \neg x_{i-1}) and p(x_i | x_{i-1})
        self.param = nn.Parameter(torch.rand(self.num_vars,2).to(device), requires_grad=True)
        self.device = device

        assert(self.chain_len**2 == self.num_vars)

        self.chainmask0 = torch.tensor([[i != term[0] for i in range(self.num_vars)] for term in chain_terms]).to(self.device) # (num_terms, num_vars)
        self.chainmask1 = torch.tensor([[i != term[1] for i in range(self.num_vars)] for term in chain_terms]).to(self.device) # (num_terms, num_vars)
        self.hopmask = torch.tensor([[(i not in term) for i in range(self.num_vars)] for term in hop_terms]).to(self.device) # (num_terms, num_vars)

    def p(self): # sigmoid of params
        return torch.clamp(torch.sigmoid(self.param), min=1e-6, max=1-(1e-6))

    def mean(self):
        if self.mean_tmp is not None: return self.mean_tmp
        ret = torch.zeros(self.chain_len, self.chain_len).to(self.device)
        p = self.p().view(self.chain_len, self.chain_len, 2)
        for i in range(self.chain_len):   # 0 indexed
            if i % self.chain_len == 0:
                ret[:,i] = p[:,i,0]
            else:
                tmp = ret[:, i-1].clone()
                ret[:,i] = (1-tmp) * p[:,i,0] + tmp * p[:,i,1]
        #print(p)
        #print(torch.tensor(ret).to(self.device))
        return ret.view(-1)

    def expectation_of_chain_terms(self, terms): # terms : (num_terms, len(term))
        num_terms = len(terms)
        #mask0 = torch.tensor([[i != term[0] for i in range(self.num_vars)] for term in terms]).to(self.device) # (num_terms, num_vars)
        #mask1 = torch.tensor([[i != term[1] for i in range(self.num_vars)] for term in terms]).to(self.device) # (num_terms, num_vars)
        mask0 = self.chainmask0
        mask1 = self.chainmask1
        p0 = self.p()[:,0].repeat(num_terms, 1)
        p1 = self.p()[:,1].repeat(num_terms, 1)
        m = self.mean().repeat(num_terms, 1)

        tmp = m.clone()
        m[~mask1] = tmp[~mask0] * p1[~mask1] + (1-tmp[~mask0]) * (1-p0[~mask1])
        m[mask1] = 1.0
        #print("chain: ", m)
        ret = torch.prod(2*m - 1, dim=-1)
        return ret

    def expectation_of_hop_terms(self, terms): # terms : (num_terms, len(term))
        num_terms = len(terms)
        #mask = torch.tensor([[(i not in term) for i in range(self.num_vars)] for term in terms]).to(self.device) # (num_terms, num_vars)
        mask = self.hopmask
        m = self.mean().repeat(num_terms, 1)
        m[mask] = 1.0

        ret = torch.prod(2*m - 1, dim=-1)
        return ret

    def entropy(self):
        self.mean_tmp = None
        ent = torch.zeros(self.chain_len).to(self.device)
        ret = torch.zeros(self.chain_len, self.chain_len).to(self.device)
        p = self.p().view(self.chain_len, self.chain_len, 2)
        for i in range(self.chain_len):   # 0 indexed
            if i % self.chain_len == 0:
                ret[:,i] = p[:,i,0]
                ent += -1 * (p[:,i,0] * torch.log(p[:,i,0]) + (1-p[:,i,0]) * torch.log(1-p[:,i,0]))
            else:
                tmp = ret[:, i-1].clone()
                ret[:,i] = (1-tmp) * p[:,i,0] + tmp * p[:,i,1]
                ent += -1 * (1-ret[:,i-1]) * (p[:,i,0] * torch.log(p[:,i,0]) + (1-p[:,i,0]) * torch.log(1-p[:,i,0]))
                ent += -1 * ret[:,i-1] * (p[:,i,1] * torch.log(p[:,i,1]) + (1-p[:,i,1]) * torch.log(1-p[:,i,1]))
        self.mean_tmp = ret.view(-1)
        return ent.sum()
