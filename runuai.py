import warnings
import argparse

import numpy as np
import torch

from UAI.graphical_model import GraphicalModel, Factor
from inference import variational_inference_mean_field, variational_inference_spn, run_mcmc, variational_inference_structured_mean_field_uai

def readModel(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        vs, cs = int(lines[1]), int(lines[3])
        lines = lines[4:]
        lines = [line for line in lines if line != "\n"]

        print(filename, vs, cs)
        print(len(lines), 3*cs)
        assert(len(lines) == 3*cs)

        varlist = lines[:cs]
        cptlist = lines[cs+1::2]
        factors = []

        for i in range(cs):
            vl = list(map(int, varlist[i].split()))[1:]
            cpt = list(map(float, cptlist[i].split()))

            n = len(vl)
            assert(len(cpt) == 2**n)
            for j in range(2**n):
                variables = [ -(vl[k]+1) if bool((2**(n-1-k)) & j) else vl[k]+1 for k in range(n) ]
                logcoeff = np.log(cpt[j])
                factors.append(Factor(logcoeff, variables))

    return GraphicalModel(num_vars=vs, factors=factors)


def run_one_network(network, timelimit, repeat, lr, outfile):
    uai_file = 'UAI/PR_prob/' + network +'.uai'
    gm = readModel(uai_file)

    pr_sol_file = 'UAI/PR_sol/' + network + '.uai.PR'
    pr_sol = readSol(pr_sol_file)
    print("Network: %s, Sol: %f" % (network, pr_sol))

    mf_best, smf_best, spn_best, mcmc_best = float("-inf"), float("-inf"), float("-inf"), float("-inf")
    task = "uai"
    for i in range(repeat):
        mf_best = max(mf_best, variational_inference_mean_field(gm, timelimit=timelimit/repeat, lr=lr, start_positive=False, device=DEVICE, task=task))
        smf_best = max(smf_best, variational_inference_structured_mean_field_uai(gm, timelimit=timelimit/repeat, lr=lr, device=DEVICE))
        spn_best = max(spn_best, variational_inference_spn(gm, timelimit=timelimit/repeat, lr=lr, spn_copies=args.spncopies, device=DEVICE, task=task))

    mcmc_best = run_mcmc(gm, timelimit=timelimit, device=DEVICE, task=task)

    result_str = "%s %f %f %f %f %f\n" % (network, pr_sol, mcmc_best, mf_best, spn_best, smf_best)
    print(result_str)
    with open(outfile, 'a') as f:
        f.write(result_str)

def readSol(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        base10_sol = float(lines[1].strip())
    return base10_sol * np.log(10)

NETWORKS = [
    'Alchemy_11',
    'DBN_11',
    'DBN_12',
    'DBN_13',
    'DBN_14',
    'DBN_15',
    'DBN_16',
    'grid10x10.f10',
    'Grids_11',
    'Grids_12',
    'Grids_13',
    'Grids_14',
    'Grids_15',
    'Grids_16',
    'Grids_17',
    'Grids_18',
    #'relational_1',     # use spn_copies=2
    'Segmentation_11',
    'Segmentation_12',
    'Segmentation_13',
    'Segmentation_14',
    'Segmentation_15',
    'Segmentation_16',
]

warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--run',     type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--net',    type=str, default="",     help="Name of network. Empty string to run all networks.")
parser.add_argument('--tl',    type=int, default=30,     help="Timelimit in minutes.")
parser.add_argument('--repeat',    type=int, default=5,     help="Number of times to restart inference.")
parser.add_argument('--lr',    type=float, default=5e-2,     help="Learning rate")
parser.add_argument('--spncopies',    type=int, default=32,     help="Spn copies")
args = parser.parse_args()

OUTFILE = "UAI/results/results_net=%s_repeat=%u_tl=%u_run=%u_copies=%u.txt" % (args.net, args.repeat, args.tl, args.run, args.spncopies)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE, OUTFILE, flush=True)

if __name__ == "__main__":
    with open(OUTFILE, 'w') as f:
        f.write("network partition mcmc mf spn smf\n")

    networks = NETWORKS
    if args.net != "": networks = [args.net]

    for net in networks:
        try:
            run_one_network(network=net, timelimit=args.tl, repeat=args.repeat, lr=args.lr, outfile=OUTFILE)
        except Exception as e:
            print(e)