import warnings
import argparse
import time

import numpy as np
from scipy.special import logsumexp
import torch
from torch import nn, optim
from tqdm import tqdm

from Ising.mean_field import MeanField as Ising_MeanField
from Ising.mcmc import MCMC as Ising_MCMC
from Ising.structured_mean_field import StructuredMeanField as Ising_StructuredMeanField

from UAI.mean_field import MeanField as UAI_MeanField
from UAI.mcmc import MCMC as UAI_MCMC
from UAI.structured_mean_field import StructuredMeanField as UAI_StructuredMeanField

from codebase.models.spns import spn
from codebase.models.spns import spn_utils as spn_ut

DISABLE_TQDM = True     # suppress tqdm output

def variational_inference_mean_field(gm, timelimit, lr, start_positive, device, task):
    # task has to be one of "uai" or "ising"
    assert(task == "uai" or task == "ising")

    def elbo_mean_field(q, terms, coeffs):
        qq = q.expectation_of_terms(terms).view(-1)
        energy = torch.dot(qq, coeffs)
        ent = q.entropy()

        elbo = energy + ent
        return elbo

    p = torch.rand(gm.num_vars)
    if start_positive: p = torch.zeros(gm.num_vars) + 3

    if task == "ising": mf = Ising_MeanField(p, device)
    if task == "uai":   mf = UAI_MeanField(p, device)
    terms = [f.variables for f in gm.factors]
    coeffs = torch.tensor([f.coefficient for f in gm.factors]).to(device)

    optimizer = optim.Adam(mf.parameters(), lr=lr)

    best_elbo = float("-inf")
    start_time = time.time()
    with tqdm(total=float("inf"), desc="Iterations", disable=DISABLE_TQDM) as pbar:
        while True:
            nelbo = -1 * elbo_mean_field(mf, terms, coeffs)
            optimizer.zero_grad()
            nelbo.backward()
            optimizer.step()

            pbar.set_postfix(elbo='{:f}'.format(-1*nelbo.data)),
            pbar.update(1)
            best_elbo = max(best_elbo, -1*nelbo.data)

            if time.time() - start_time > 60*timelimit:
                break

    return best_elbo

def variational_inference_spn(gm, timelimit, lr, spn_copies, device, task):
    # task has to be one of "uai" or "ising"
    assert(task == "uai" or task == "ising")

    def elbo_spn(q, terms, coeffs, num_vars):
        if task == "ising": qq = q.expectation_of_terms_ising(terms).view(-1)
        if task == "uai":   qq = q.expectation_of_terms_uai(terms).view(-1)
        energy = torch.dot(qq, coeffs)
        ent = q.entropy_selective()
        ent -= (spn.num_vars - num_vars) * np.log(2)     # number of variables in spn is a power of 2, so there may be some padding variables that we have to subtract.
        
        elbo = energy + ent
        return elbo

    spn = spn_ut.construct_spn_structure(
                num_vars=gm.num_vars,
                max_copies=spn_copies,
                device=device) # spn structure
    # print(spn.num_params())
    print(spn.param_shape_per_layer)

    terms = [f.variables for f in gm.factors]
    coeffs = torch.tensor([f.coefficient for f in gm.factors]).to(device)

    optimizer = optim.Adam(spn.parameters(), lr=lr)

    best_elbo = float("-inf")
    start_time = time.time()
    with tqdm(total=float("inf"), desc="Iterations", disable=DISABLE_TQDM) as pbar:
        while True:
            nelbo = -1 * elbo_spn(spn, terms, coeffs, gm.num_vars)[0][0]
            optimizer.zero_grad()
            nelbo.backward()
            optimizer.step()

            pbar.set_postfix(elbo='{:f}'.format(-1*nelbo.data))
            pbar.update(1)
            best_elbo = max(best_elbo, -1*nelbo.data)

            if time.time() - start_time > 60*timelimit:
                break

    return best_elbo

def run_mcmc(gm, timelimit, device, task):
    # task has to be one of "uai" or "ising"
    assert(task == "uai" or task == "ising")

    terms = [f.variables for f in gm.factors]
    coeffs = [f.coefficient for f in gm.factors]
    if task == "ising": mcmc = Ising_MCMC(gm.num_vars, coeffs, terms, device)
    if task == "uai":   mcmc = UAI_MCMC(gm.num_vars, coeffs, terms, device)

    log_sum = float("-inf")
    estimate = log_sum
    start_time = time.time()
    with tqdm(total=float("inf"), desc="Iterations", disable=DISABLE_TQDM) as pbar:
        it = 0
        while True:
            density = mcmc.sample()
            log_sum = logsumexp([log_sum, density.cpu().detach().numpy()])

            estimate = log_sum - np.log(it+1)
            pbar.set_postfix(estimate='{:f}'.format( estimate ))
            pbar.update(1)
            it += 1

            if time.time() - start_time > 60*timelimit:
                break

    return estimate

def variational_inference_structured_mean_field_ising(gm, timelimit, lr, device, gridsize):
    def elbo(q, chain_terms, chain_coeffs, hop_terms, hop_coeffs):    
        ent = q.entropy()
        qq = q.expectation_of_chain_terms(chain_terms).view(-1)
        energy = torch.dot(qq, chain_coeffs)
        qq = q.expectation_of_hop_terms(hop_terms).view(-1)
        energy += torch.dot(qq, hop_coeffs)
        
        elbo = energy + ent
        return elbo
    
    chain_factors = [f for f in gm.factors if f.variables[0]+1 == f.variables[1]]
    hop_factors = [f for f in gm.factors if f.variables[0]+1 != f.variables[1]]
    chain_terms = [f.variables for f in chain_factors]
    chain_coeffs = torch.tensor([f.coefficient for f in chain_factors]).to(device)
    hop_terms = [f.variables for f in hop_factors]
    hop_coeffs = torch.tensor([f.coefficient for f in hop_factors]).to(device)

    smf = Ising_StructuredMeanField(gm.num_vars, gridsize, chain_terms, hop_terms, device)
    #print(chain_terms, hop_terms)

    optimizer = optim.Adam(smf.parameters(), lr=lr)

    best_elbo = float("-inf")
    start_time = time.time()
    with tqdm(total=float("inf"), desc="Iterations", disable=DISABLE_TQDM) as pbar:
        while True:
            nelbo = -1 * elbo(smf, chain_terms, chain_coeffs, hop_terms, hop_coeffs)
            optimizer.zero_grad()
            nelbo.backward()
            optimizer.step()

            pbar.set_postfix(elbo='{:f}'.format(-1*nelbo.data))
            pbar.update(1)
            best_elbo = max(best_elbo, -1*nelbo.data)

            if time.time() - start_time > 60*timelimit:
                break

    return best_elbo

def variational_inference_structured_mean_field_uai(gm, timelimit, lr, device):
    def elbo(q, terms, coeffs):    
        ent = q.entropy()
        qq = q.expectation_of_terms().view(-1)
        energy = torch.dot(qq, coeffs)
        
        elbo = energy + ent
        return elbo

    terms = [f.variables for f in gm.factors]
    coeffs = torch.tensor([f.coefficient for f in gm.factors]).to(device)
    smf = UAI_StructuredMeanField(gm.num_vars, terms, device)

    optimizer = optim.Adam(smf.parameters(), lr=lr)

    best_elbo = float("-inf")
    start_time = time.time()
    with tqdm(total=float("inf"), desc="Iterations", disable=DISABLE_TQDM) as pbar:
        while True:
            nelbo = -1 * elbo(smf, terms, coeffs)
            optimizer.zero_grad()
            nelbo.backward()
            optimizer.step()

            pbar.set_postfix(elbo='{:f}'.format(-1*nelbo.data))
            pbar.update(1)
            best_elbo = max(best_elbo, -1*nelbo.data)

            if time.time() - start_time > 60*timelimit:
                break

    return best_elbo