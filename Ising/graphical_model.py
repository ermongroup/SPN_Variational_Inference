from operator import mul
from scipy.special import logsumexp
import math

# Each factor has form c l_1 l_2 ... l_k
# Factor evalutes to c if number of unsatisfied literals is even
# Otherwise factor evalutes to -c

class GraphicalModel():
    def __init__(self, num_vars, factors):
        self.num_vars = num_vars    # should be a square number since we have a grid
        self.factors = factors      # array of Factors

    def log_density(self, evid):
        # return the unnormalized log density of log p(evid)
        return sum([factor.eval_evidence(evid) for factor in self.factors])

    def exact_log_partition(self):
        if self.num_vars > 16:  return 0    # skip if too computationally intensive
        log_partition = float("-inf")

        mx = 1 << (self.num_vars)
        for mask in range(mx):
            evid = [(i, bool(mask & (1<<i) ) ) for i in range(self.num_vars)]
            evid = dict([(k, v*2 - 1) for k,v in evid]) # convert v from {0,1} to {-1,1}, and turn into dict

            log_partition = logsumexp([log_partition, self.log_density(evid)])

        return log_partition

    def write_to_file(self, file):
        with open(file, 'w') as f:
            f.write("MARKOV\n")
            f.write("%u\n" % self.num_vars)
            f.write("2 " * self.num_vars)
            f.write("\n")

            f.write("%u\n" % len(self.factors))
            for factor in self.factors:
                f.write("2 " +  " ".join(list(map(str, factor.variables))) + "\n")
            for factor in self.factors:
                expcoeff = math.exp(factor.coefficient)
                nexpcoeff = math.exp(-1*factor.coefficient)
                f.write("4\n")
                f.write("%f %f %f %f \n" % (expcoeff, nexpcoeff, nexpcoeff, expcoeff))

    def read_from_file(self, grid, file):
        print("load %s" % file)
        with open(file, 'r') as f:
            lines = f.readlines()
            lines = lines[4:]
            lines = [line for line in lines if line != "\n"]

            self.num_vars = grid*grid
            cs = grid * (grid-1) * 2
            self.factors = [None for _ in range(cs)]

            print(len(lines), cs)
            assert(len(lines) == 3*cs)

            varlist = lines[:cs]
            cptlist = lines[cs+1::2]

            for i in range(cs):
                vl = list(map(int, varlist[i].split()))[1:]
                cpt = list(map(float, cptlist[i].split()))
                cpt = [math.log(x) for x in cpt]

                self.factors[i] = Factor(cpt[0], vl)

class Factor():
    def __init__(self, coefficient, variables):
        # coefficient:  float("-inf") or any real number
        # variables:    a list of pos integers in the range [0, num_vars)
        self.coefficient = float(coefficient)
        self.variables = variables

    def eval_evidence(self, evid):
        # only eval full evidence
        ret = self.coefficient
        for k, v in evid.items():
            if abs(k) in self.variables:
                ret *= 1 if (v > 0) else -1
        return ret
