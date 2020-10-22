from operator import mul
from scipy.special import logsumexp

# Following Uai format: http://www.hlt.utdallas.edu/~vgogate/uai16-evaluation/binaryformat.html
# Each factor has form c l_1 l_2 ... l_k
# Factor evalutes to 0 if one of the literals is satisfied
# Otherwise factor evalutes to c

class GraphicalModel():
    def __init__(self, num_vars, factors):
        self.num_vars = num_vars    # should be even after conversion from uai to buai
        self.factors = factors      # array of Factors

    def log_density(self, evid):
        # return the unnormalized log density of log p(evid)
        return sum([factor.eval_evidence(evid) for factor in self.factors])

    def exact_log_partition(self):
        if self.num_vars > 16:  return 0    # skip if too computationally intensive
        log_partition = float("-inf")

        mx = 1 << (self.num_vars)
        for mask in range(mx):
            evid = [(i+1, bool(mask & (1<<i) ) ) for i in range(self.num_vars)]
            evid = dict([(k, v*2 - 1) for k,v in evid]) # convert v from {0,1} to {-1,1}, and turn into dict

            log_partition = logsumexp([log_partition, self.log_density(evid)])

        return log_partition


class Factor():
    def __init__(self, coefficient, variables):
        # coefficient:  float("-inf") or any real number
        # variables:    a list of pos/neg integers in the range [-num_vars, -1] u [1, num_vars]
        self.coefficient = float(coefficient)
        self.variables = variables

    def eval_evidence(self, evid):
        eval_true = False

        for k, v in evid.items():
            if k in self.variables:     eval_true |= (v > 0)
            if -1*k in self.variables:  eval_true |= (v < 0)

        return 0 if eval_true else self.coefficient
