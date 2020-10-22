# Third party imports
import numpy as np
import torch
from torch import nn

# Local application imports
from codebase.models.spns.spn import SPN

def construct_spn_structure(num_vars, max_copies, device, num_roots=1, normalize=True):
    num_vars = 2**((num_vars - 1).bit_length()) # round to next power of 2
    num_leaves = 2 * num_vars # +ve and -ve for each var
    # print("# leaves: ", num_leaves)

    leaf_to_var = np.repeat(np.arange(num_vars), 2)
    # print("leaf_to_var: ", leaf_to_var)

    spn = SPN(num_vars=num_vars, device=device, normalize=normalize)
    spn.add_bernoulli_layer(num_leaves, leaf_to_var)
    
    # number of product layers = log(num_vars)
    # product layers reduce vars_partitions by factor of 2
    # sum layers: form deterministic partitions
    num_product_layers = (num_vars-1).bit_length()
    partitions, copies = num_vars, 2

    for i in range(num_product_layers):
        MAX_C = max_copies # power of 2
        if copies > MAX_C:
            new_copies = MAX_C
            spn.add_sum_layer(partitions * new_copies)
            copies = new_copies

        new_partitions = partitions//2
        new_copies = copies ** 2

        # cartesian product between copies of two children vars_partition
        spn.add_product_layer(new_partitions*new_copies, new_copies, new_partitions)
        copies = new_copies
        partitions = new_partitions

    assert(partitions == 1)
    if copies != num_roots: # copies > num_roots and copies < num_roots are both possible
        new_copies = num_roots
        spn.add_sum_layer(new_copies)

    return spn.ready()