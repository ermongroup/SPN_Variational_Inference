# Third party imports
import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.autograd import Variable as Variable

# Local application imports
from codebase.models.spns import spn_nodes as Nodes

EPS = 1e-15

class SPN(torch.nn.Module):
    def __init__(self, num_vars, device, normalize=True):
        super(SPN, self).__init__()
        self.num_vars = num_vars # number of variables of the SPN
        self.device = device
        self.normalize = normalize

        self.leaf_layer = None # Layer for leaf distributions.
        self.layers = [] # 1D array of layers
        self.net = None # nn.Sequential module, called once self.layers is ready

        self.param_shape_per_layer = []

        self.terms_mask = None

    def ready(self):
        self.leaf_layer.initialize()
        for layer in self.layers:
            layer.initialize()

        self.num_roots = self.layers[-1].num
        self.net = nn.ModuleList([self.leaf_layer] + self.layers)

        return self

    def num_params(self):
        return sum([np.prod(p) for p in self.param_shape_per_layer])

    def reinitialize(self, params):
        # params: (self.num_params())
        assert(params.size() == torch.Size([self.num_params()]))

        leaf_params_count = np.prod(self.param_shape_per_layer[0])
        leaf_params, params = params[:leaf_params_count], params[leaf_params_count:]

        # leaf layer
        if isinstance(self.leaf_layer, Nodes.BernoulliLayer):
            self.leaf_layer.initialize(params=leaf_params)
        else:
            assert(False)

        # inner layers
        for i,layer in enumerate(self.layers):
            params_count = np.prod(self.param_shape_per_layer[i+1])
            layer_params, params = params[:params_count], params[params_count:]
            layer_params = layer_params.view(*self.param_shape_per_layer[i+1])
            layer.initialize(layer_params)

        assert(torch.numel(params) == 0)

    def expectation_of_terms_uai(self, terms): # careful: this version returns probability terms are unsatisfied
        # terms: (num_terms, num_vars)
        batch = len(terms)

        if self.terms_mask is None:
            self.terms_mask = torch.tensor([[ (i in term) - (-1*i in term)   for i in range(1,self.num_vars+1)] for term in terms]).to(self.device) # (num_terms, num_vars)
            self.terms_mask = self.terms_mask.view(batch, self.num_vars, 1).repeat(1, 1, 2)
            self.terms_mask = self.terms_mask.view(batch,self.leaf_layer.num).repeat(1, 1) # pattern of last dim: [0,0,1,1,2,2,...,n,n]
        mask = self.terms_mask

        p = self.leaf_layer.get_pparams().unsqueeze(0).repeat(batch,1).to(self.device) # (batch, self.leaf_layer.num)

        unsat_condition = (((p > 0) != (mask > 0)) + (mask == 0)).bool()
        output = (unsat_condition).float().to(self.device)

        output = torch.clamp(output, min=EPS, max=1-EPS)
        output = torch.log(output)

        for layer in self.layers:
            output = layer.forward(output)

        output = torch.exp(output)
        return output

    def expectation_of_terms_ising(self, terms):
        # terms: (num_terms, num_vars)
        batch = len(terms)

        if self.terms_mask is None:
            self.terms_mask = torch.tensor([[(i not in term) for i in range(self.num_vars)] for term in terms])
        mask = self.terms_mask
        mask = mask.view(batch, self.num_vars, 1).repeat(1, 1, 2)
        #mask = torch.tensor([(i not in term) for i in range(self.num_vars)])
        mask = mask.view(batch,self.leaf_layer.num)
        p = self.leaf_layer.get_pparams().unsqueeze(0).repeat(batch,1) # (batch, self.leaf_layer.num)

        output = (2*p-1)
        output[mask] = 1.0

        for layer in self.layers:
            output = layer.forward_no_log(output)
        return output

    def add_sum_layer(self, num):
        """
        Adds a sum-node layer

        Args:
            num: int: (): number of sum nodes in the layer
        """
        child_num = self.layers[-1].num if self.layers else self.leaf_layer.num
        self.param_shape_per_layer.append(child_num)
        self.layers.append(Nodes.SumLayer(num=num, child_num=child_num, device=self.device, normalize=self.normalize))
        return self.layers[-1]

    def add_product_layer(self, num, copies, partitions):
        """
        Adds a product-node layer

        Args:
            num: int: (): number of product nodes in the layer
            edges: np_arr: (num_children, num): edge connection from this layer to the child layer
        """
        self.param_shape_per_layer.append([0])
        self.layers.append(Nodes.ProductLayer(num=num, copies=copies, partitions=partitions, device=self.device))
        return self.layers[-1]

    def add_bernoulli_layer(self, num, var):
        """
        Adds a Bernoulli leaf layer

        Args:
            num: int: (): number of nodes in the layer
            var: np_arr: (num): scope variables of the leaf nodes
        """
        var_param = torch.from_numpy(var).to(self.device)
        self.param_shape_per_layer.append([num])
        self.leaf_layer = Nodes.BernoulliLayer(num=num, var=var_param, device=self.device)
        return self.leaf_layer

    def entropy_selective(self):
        """
        Computes the entropy of the spn exactly, if the spn is selective.

        Return:
            ent: tensor: (spn_copies, 1, root.num): The entropy of the spn
        """
        ent = self.leaf_layer.entropy()

        for layer in self.layers:
            ent = layer.entropy(ent)
        return ent