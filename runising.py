import warnings
import argparse

import numpy as np
import torch

from Ising.graphical_model import GraphicalModel, Factor
from inference import variational_inference_mean_field, variational_inference_spn, run_mcmc, variational_inference_structured_mean_field_ising

def generate_random_factors_grid(num_var, pairwise_strength, single_strength):
    import math, random

    factors = []
    side = int(math.sqrt(num_var))
    for i in range(num_var):
        for j in range(i+1, num_var):
            row_i, col_i, row_j, col_j = i//side, i%side, j//side, j%side
            # if not adjacent, continue
            if abs(row_i - row_j) + abs(col_i - col_j) > 1:
                continue

            w = random.uniform(0, 1)
            if args.mode == 1:
                w = 2*w - 1
            elif args.mode == 2:
                pass
            else:
                assert(False)

            w *= pairwise_strength
            factors.append(Factor(w, [i, j]))

    return factors


def run_one_network(num_var, pairwise_strength, single_strength, timelimit, repeat, lr, outfile):
    gm_file = GM_PREFIX + "_%.0f.uai" % pairwise_strength
    if args.loadgm < 0:
        factors = generate_random_factors_grid(num_var, pairwise_strength, single_strength)
        gm = GraphicalModel(num_var, factors)
        gm.write_to_file(gm_file)
    else:
        gm = GraphicalModel(0, [])
        gm.read_from_file(args.n, gm_file)

    partition_fn = gm.exact_log_partition()
    print(partition_fn)

    mf_best, smf_best, spn_best, mcmc_best = float("-inf"), float("-inf"), float("-inf"), float("-inf")
    task = "ising"
    for z in range(repeat):
        start_positive = (args.mode == 2) and (z%2)  # try positive initialization
        mf_best = max(mf_best, variational_inference_mean_field(gm, timelimit=timelimit/repeat, lr=lr, start_positive=start_positive, device=DEVICE, task=task))
        smf_best = max(smf_best, variational_inference_structured_mean_field_ising(gm, timelimit=timelimit/repeat, lr=lr, device=DEVICE, gridsize=args.n))
        spn_best = max(spn_best, variational_inference_spn(gm, timelimit=timelimit/repeat, lr=lr, spn_copies=args.spncopies, device=DEVICE, task=task))

    if args.n < 32: mcmc_best = run_mcmc(gm, timelimit=timelimit, device=DEVICE, task=task)

    result_str = "%f %f %f %f %f %f\n" % (pairwise_strength, partition_fn, mcmc_best, mf_best, spn_best, smf_best)
    print(result_str)
    with open(outfile, 'a') as f:
        f.write(result_str)


warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--n',      type=int, default=4,     help="Constructs grid of size n x n")
parser.add_argument('--run',     type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--mode',    type=int, default=1,     help="1: mixed, 2: positive")
parser.add_argument('--tl',    type=int, default=30,     help="number of minutes for each inference run")
parser.add_argument('--repeat',    type=int, default=5,     help="Number of times to restart inference.")
parser.add_argument('--lr',    type=float, default=5e-2,     help="Learning rate")
parser.add_argument('--loadgm',    type=int, default=-1,     help="If -1, create new ising grids with ID=args.run. Otherwise, load ising grids from ID=args.loadgm")
parser.add_argument('--spncopies',    type=int, default=32,     help="Spn copies")
args = parser.parse_args()

loadid = args.run if args.loadgm < 0 else args.loadgm
OUTFILE = "Ising/results/results_n=%u_mode=%s_repeat=%u_tl=%u_run=%u_copies=%u.txt" % (args.n, args.mode, args.repeat, args.tl, args.run, args.spncopies)
GM_PREFIX = "Ising/models/models_%u_%s_%s" % (args.n, args.mode, loadid)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE, OUTFILE, flush=True)

if __name__ == "__main__":
    with open(OUTFILE, 'w') as f:
        f.write("strength partition mcmc mf spn smf\n")

    num_var = args.n*args.n # must be power of 2

    for pairwise_strength in range(2,16,2):
        print(pairwise_strength)
        run_one_network(num_var, float(pairwise_strength), single_strength=0.0, timelimit=args.tl, repeat=args.repeat, lr=args.lr, outfile=OUTFILE)
