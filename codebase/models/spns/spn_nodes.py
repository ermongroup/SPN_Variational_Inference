# Standard libaray imports
from abc import ABC, abstractmethod

# Third party imports
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable as Variable

EPS = 1e-15

class NodeLayer(torch.nn.Module, ABC):
    def __init__(self):
        super(NodeLayer, self).__init__()

    @abstractmethod
    def initialize(self):
        pass

class SumLayer(NodeLayer):
    def __init__(self, num, child_num, device, normalize=True):
        """
        :param num: the number of sum nodes in this layer
        :param child_num: the number of sum nodes in the last layer
        """
        NodeLayer.__init__(self).__init__()
        self.num = num
        self.child_num = child_num
        self.device = device
        self.normalize = normalize

        self.child_per_node = self.child_num // self.num

    def initialize(self, params=None):
        if params:
            self.logparams = params
        else:
            self.logparams = nn.Parameter(torch.log(torch.ones([self.child_num]).uniform_() / self.child_per_node).to(self.device))

    def forward_no_log(self, input):
        """
        Args:
            input: tensor: (batch, child.num): the input to the layer

        Return:
            node_output: tensor: (batch, self.num): the output of nodes in this layer
        """
        batch, _ = input.size()

        input = input.view(batch, self.num, self.child_per_node)
        logparams = self.get_logparams()
        params = torch.exp(logparams)

        node_output = (input * params).sum(dim=-1)
        return node_output


    def forward(self, input):
        """
        Args:
            input: tensor: (batch, child.num): the input to the layer

        Return:
            node_output: tensor: (batch, self.num): the output of nodes in this layer
        """
        batch, _ = input.size()

        input = input.view(batch, self.num, self.child_per_node)
        params = self.get_params()
        logparams = torch.log(params)

        node_output = torch.logsumexp(input + logparams, dim=-1)
        return node_output

    def entropy(self, input):
        params = self.get_params()
        logparams = torch.log(params)
        wt_sum = self.forward_no_log(input)
        wlogw = (params * logparams).sum(dim=-1).unsqueeze(0) # unsqueeze batch dim
        return wt_sum - wlogw

    def get_params(self):
        # logparams shape (self.num, C)
        params = torch.exp(self.logparams).view(self.num, self.child_per_node)
        if self.normalize: params = params / torch.sum(params, dim=-1, keepdims=True)

        return params

    def get_logparams(self):
        # logparams shape (self.num, C)
        return torch.log(self.get_params() + EPS)

class ProductLayer(NodeLayer):
    def __init__(self, num, copies, partitions, device):
        """
        :param num: the number of product nodes in this layer
        :param copies: the number of copies of each partition
        :param partitions: number of variable partitions in this layer
        invariant: num = copies x partitions
        """
        NodeLayer.__init__(self).__init__()
        self.num = num
        self.copies = copies
        self.partitions = partitions
        assert(self.num == self.copies * self.partitions)
        self.device = device

        self.ch_copies = np.round(np.sqrt(self.copies)).astype(int)

        y = torch.arange(self.num).to(self.device)
        group = y // (self.copies)
        offset = y % (self.copies)
        self.ch1 = group*(2*self.ch_copies) + offset // self.ch_copies
        self.ch2 = group*(2*self.ch_copies) + self.ch_copies + offset % self.ch_copies

    def initialize(self, params=None):
        pass

    def forward(self, input):
        """
        Args:
            input: tensor: (batch, child.num): the input to the layer

        Return:
            node_output: tensor: (batch, self.num): the output to the layer
        """
        node_output = input[:,self.ch1] + input[:,self.ch2]
        return node_output

    def forward_no_log(self, input):
        """
        Args:
            input: tensor: (batch, child.num): the input to the layer

        Return:
            node_output: tensor: (batch, self.num): the output to the layer
        """
        node_output = input[:,self.ch1] * input[:,self.ch2]
        return node_output

    def entropy(self, input):
        return self.forward(input)

class BernoulliLayer(NodeLayer):
    def __init__(self, num, var, device):
        """
        A leaf layer made up of Bernoulli nodes.

        Args:
            num: tensor: (): the number of leaf nodes
            var: tensor: (self.num): scope variable of the nodes (0-indexed)
            p: tensor: (self.num): prob of the Bernoulli nodes being True
            device: CPU or GPU
        """
        NodeLayer.__init__(self).__init__()
        self.num = num
        self.var = var
        self.device = device

    def initialize(self, params=None):
        if params:
            self.p = params
        else:
            self.p = torch.tensor([0.,1.]).repeat(self.num // 2).to(self.device)

    def entropy(self):
        return torch.zeros((1,self.p.size(0))).to(self.device)

    def get_pparams(self):
        return self.p