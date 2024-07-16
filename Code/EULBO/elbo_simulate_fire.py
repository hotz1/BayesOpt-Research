import math
import numpy as np
import torch
STD_normal = torch.distributions.normal.Normal(0, 1) # Standard Normal

from typing import Optional, Tuple
# Type hints are strictly optional, but personally I find that they make code more reasonable

import jaxtyping
from jaxtyping import Float, Integer
from collections.abc import Callable
# This package allows type annotations that include the size of torch Tensors/numpy arrays
# It's not necessary, but it helps with understanding what each function does

from torch import Tensor

# import tqdm.notebook as tqdm
# from tqdm import tqdm
import pandas as pd
import os as oper
import fnmatch
import time
import fire

# Set DTYPE and DEVICE variables for torch tensors
DTYPE = torch.float32
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def mu(X: Float[Tensor, "N D"]) -> Float[Tensor, "N"]:
    r"""
    Computes the (very lame) zero mean function mu(X) = 0
    """

    return torch.zeros(*X.shape[:-1], dtype=X.dtype, device=X.device)

def matern_kernel(
    X1: Float[Tensor, "M D"], 
    X2: Float[Tensor, "N D"],
    ls: Float[Tensor, "1 D"], 
    os: Float[Tensor, "1 1"],
) -> Float[Tensor, "M N"]:
    r"""
    Computes Matern 5/2 kernel across all pairs of points (rows) in X1 & X2

    k(X1, X2) = os * (1 + \sqrt{5} * D + 5/3 * (D**2)) * exp(-\sqrt{5} * D)
    D = || (X1 - X2) / ls ||_2
    https://en.wikipedia.org/wiki/Mat%C3%A9rn_covariance_function

    ls: lengthscale
    os: outputscale
    """

    # Compute D, D ** 2, \sqrt{5} * D
    D_sq = (X1.div(ls).unsqueeze(-2) - X2.div(ls).unsqueeze(-3)).square().sum(dim = -1)
    
    D = torch.sqrt(D_sq + 1e-20)  # The 1e-20 is for numerical stability, so we don't get any NaNs if D≈0 but is very small and negative
    
    # Compute and return kernel
    return torch.mul(
        1 + (math.sqrt(5) * D) + ((5. / 3) * D_sq),
        torch.exp(-math.sqrt(5) * D)
    ).mul(os)

def compute_posterior_mean_and_variance(
    test_inputs: Float[Tensor, "M D"],
    X: Float[Tensor, "N D"],
    y: Float[Tensor, "N"],
    K_chol: Float[Tensor, "N N"],
    ls: Float[Tensor, "1 D"],
    os: Float[Tensor, "1 1"],
) -> tuple[Float[Tensor, "M"], Float[Tensor, "M M"]]:
    r"""
    Given inputs where we will evaluate the posterior, computes and returns the posterior moments
    - E[ f(test_inputs) | y ] = mu(test_inputs) + k(test_inputs, X) @ k(X, X)^{-1} @ (y - mu(X))
    - Cov[ f(test_inputs) | y ] = k(test_inputs, test_inputs) + k(test_inputs, X) @ k(X, X)^{-1} @ k(X, test_inputs)

    test_inputs:     the matrix containing test inputs we want to evaluate f() on
    X:               the matrix containing training inputs (where we have observations)
    y:               is the vector of training observations
    K_chol:          the Cholesky factor of the k(X, X) kernel matrix evaluated on training inputs
                     plus observational noise
                         i.e. K_chol @ K_chol.T = (k(X, X) + sigma^2 I)
    ls:              is the lengthscale of the kernel
    os:              is the outputscale of the kernel
    """

    # Compute k(X, X)^{-1} k(X, test_inputs)
    # We need this term for both the posterior mean and posterior variance
    Ktest = matern_kernel(X, test_inputs, ls, os)
    K_inv_Ktest = torch.cholesky_solve(Ktest, K_chol, upper=False)

    # Compute posterior mean
    posterior_mean = mu(test_inputs) + (K_inv_Ktest.mT @ (y - mu(X)).unsqueeze(-1)).squeeze(-1)

    # Compute posterior covariance
    posterior_covar = matern_kernel(test_inputs, test_inputs, ls, os) - Ktest.mT @ K_inv_Ktest

    # Done!
    return posterior_mean, posterior_covar

def compute_posterior_mean_and_variance_action(
    test_inputs: Float[Tensor, "M D"],
    X: Float[Tensor, "N D"],
    y: Float[Tensor, "N"],
    S: Float[Tensor, "N T"],
    STKS_chol: Float[Tensor, "T T"],
    ls: Float[Tensor, "1 D"],
    os: Float[Tensor, "1 1"],
) -> tuple[Float[Tensor, "M"], Float[Tensor, "M M"]]:
    r"""
    Given inputs where we will evaluate the posterior, computes and returns the posterior moments
    - E[ f(test_inputs) | y ] = mu(test_inputs) + k(test_inputs, X) @ k(X, X)^{-1} @ (y - mu(X))
    - Cov[ f(test_inputs) | y ] = k(test_inputs, test_inputs) + k(test_inputs, X) @ k(X, X)^{-1} @ k(X, test_inputs)

    test_inputs:     the matrix containing test inputs we want to evaluate f() on
    X:               the matrix containing training inputs (where we have observations)
    y:               is the vector of training observations
    K_chol:          the Cholesky factor of the k(X, X) kernel matrix evaluated on training inputs
                     plus observational noise
                         i.e. K_chol @ K_chol.T = (k(X, X) + sigma^2 I)
    ls:              is the lengthscale of the kernel
    os:              is the outputscale of the kernel
    """

    # Compute (S^T k(X, X) S)^{-1} k(X, test_inputs)
    # We need this term for both the posterior mean and posterior variance
    Ktest = matern_kernel(test_inputs, X, ls, os)
    C_cholish = torch.linalg.solve_triangular(STKS_chol, S.mT, upper = False).mT
    Ktest_Ccholish = Ktest @ C_cholish

    # Compute posterior mean
    posterior_mean = mu(test_inputs) + (Ktest_Ccholish @ C_cholish.mT @ (y - mu(X)).unsqueeze(-1)).squeeze(-1)

    # Compute posterior covariance
    posterior_covar = matern_kernel(test_inputs, test_inputs, ls, os) - (Ktest_Ccholish @ Ktest_Ccholish.mT)
    
    # Done!
    return posterior_mean, posterior_covar

def update_chol_newdata(
    K_chol: Float[Tensor, "N N"],
    X: Float[Tensor, "N D"],
    X_next: Float[Tensor, "N_next D"],
    ls: Float[Tensor, "1 D"],
    os: Float[Tensor, "1 1"],
    sigma_sq: Float[Tensor, ""],
    eps: float = 1e-4,
) -> Float[Tensor, "(N+N_next) (N+N_next)"]:
    r"""
    Computes the Cholesky factor of the block matrix
    [ k(X, X) + sigma_sq * I          k(X, X_next)                      ]
    [ k(X_next, X)                    k(X_next, X_next)  + sigma_sq * I ]
    where k is the kernel covariance

    This function should efficiently use prior computation.
    Given that we already have computed K_chol @ K_chol.T = k(X, X),
    we should be able to "update" that K_chol in O(N^2) time to get the
    desired block cholesky factorization.

    K_chol: Cholesky factorization of k(X, X) + sigma_sq * I
    X: Prior data
    X_next: Newly-added data
    ls: Length scale of the kernel covariance
    os: Output scale of the kernel covariance
    sigma_sq: Observation noise
    eps:      Small amount of noise to add to the diagonal for stability
    """

    X_joint = torch.cat([X, X_next], dim = -2)
    K_joint = matern_kernel(X_joint, X_joint, ls, os)

    # Add sigma_sq * I to K_joint
    I = torch.eye(K_joint.size(-1), dtype = K_joint.dtype, device = K_joint.device)
    K_joint = K_joint + (sigma_sq + eps) * I

    # Get the sub-blocks of K_joint
    N = X.shape[0]
    N_next = X_next.shape[0]
    K_11, K_12, K_22 = K_joint[0:N, 0:N], K_joint[0:N, N:], K_joint[N:, N:]

    # Cholesky factorization on the sub-blocks
    L_21 = torch.linalg.solve_triangular(K_chol, K_12, upper = False).mT
    L_22 = torch.linalg.cholesky(K_22 - L_21 @ L_21.mT, upper = False)

    # Concatenate sub-blocks of Cholesky decomposition matrix and return them in the form
    # [ K_chol      0 ]
    # [ L_21     L_22 ]
    return torch.cat(
        (torch.cat((K_chol, torch.zeros(N, N_next)), dim = -1), 
         torch.cat((L_21, L_22), dim = -1)
        ), dim = -2)

def hartmann_six(X: Float[Tensor, "N 6"]) -> Float[Tensor, "N"]:
    r"""
    Computes the value of the Hartmann six-dimensional test function on N rows of input data
    More info on this test function at: https://www.sfu.ca/~ssurjano/hart6.html
    """

    ### TODO: Check if inputs are "valid" (possibly)
    
    alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], dtype = DTYPE, device = X.device)
    A = torch.tensor([[10, 3, 17, 3.5, 1.7, 8],
                      [0.05, 10, 17, 0.1, 8, 14],
                      [3, 3.5, 1.7, 10, 17, 8],
                      [17, 8, 0.05, 10, 0.1, 14]],
                     dtype = DTYPE, device = X.device)
    P = 1e-4 * torch.tensor([[1312, 1696, 5569, 124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091, 381]], 
                            dtype = DTYPE, device = X.device)

    # Calculate "inner sums" 
    inner_sums: Float[Tensor, "N 4"] = torch.sum(A * (X.unsqueeze(-2) - P).pow(2), -1)

    # Exponentiate and compute "outer sums"
    outer_sums: Float[Tensor, "N"] = alpha @ torch.exp(-inner_sums).mT
    
    return outer_sums

def observe(
    func: Callable[[Float[Tensor, "N D"]], Float[Tensor, "N"]],
    X: Float[Tensor, "N D"], 
    sigma_sq: Float = 1e-2,
) -> Float[Tensor, "N"]:
    r"""
    A "wrapper" to return y = func(X) + noise.

    func: A real-valued function defined on R^D which is applied row-wise to X
    X: A matrix of N D-dimensional real-valued inputs to the function
    sigma_sq: Variance of the IID observation noise 
    """
    
    true_obs = func(X)
    return true_obs + torch.randn_like(true_obs).mul(math.sqrt(sigma_sq))
    # return true_obs

def ELBO(
    S: Float[Tensor, "N T"],
    X: Float[Tensor, "N D"],
    y: Float[Tensor, "N"],
    KXX: Float[Tensor, "N N"],
    STKS_chol: Float[Tensor, "T T"],
    ls: Float[Tensor, "1 D"],
    os: Float[Tensor, "1 1"],
    sigma_sq: Float[Tensor, "1 1"]
) -> Float[Tensor, " "]:
    r"""
    A function to compute the ELBO for action matrix S based on the existing dataset D = (X, y).

    Parameters:
        S: A queried action matrix 
        X: The "input values" in the observed dataset
        y: The corresponding "outputs" for the observed dataset
        KXX: k(X, X) + sigma_sq * I
        STKS_chol: Cholesky decomposition of S'(k(X, X) + sigma_sq * I)S

    GP Hyperparameters:
        ls: length scale of inputs
        os: output scale
        sigma_sq: Variance of observation noise

    Returns
    ELBO: The ELBO corresponding to conditioning on S'X and S'y
    """

    # Get posterior of f | S'X, S'y
    post_mean, post_var = compute_posterior_mean_and_variance_action(X, X, y, S, STKS_chol, ls, os)

    # C = S (S'(K + sigma_sq * I)S)^(-1) S'
    # C = S @ torch.cholesky_solve(S.mT, STKS_chol)
    C_cholish = torch.linalg.solve_triangular(STKS_chol, S.mT, upper = False).mT

    # Add up the individual terms
    ELBO = 0.5 * (
        (y - post_mean).square().sum().div(sigma_sq) +
        post_var.trace() +
        S.shape[0] * math.log(2 * math.pi * sigma_sq) - S.shape[1] * math.log(sigma_sq) + 
        y @ C_cholish @ C_cholish.mT @ (KXX - sigma_sq * torch.eye(KXX.shape[0])) @ C_cholish @ C_cholish.mT @ y -
        torch.cholesky_solve(STKS_chol @ STKS_chol.mT - sigma_sq * S.mT @ S, STKS_chol).trace() +
        STKS_chol.diag().log().sum().mul(2) - torch.linalg.slogdet(S.mT @ S).logabsdet
    )
    return ELBO.view(1).squeeze(-1)

def normalize_cols(S: Float[Tensor, "R C"]) -> Float[Tensor, "R C"]:
    r"""
    Helper function to ensure that each column of the matrix has an L2 norm of 1 
    """

    # Calculate L2 norms of columns of the given matrix
    col_norms: Float[Tensor, "C"] = (S * S).sum(-2).sqrt()

    # Error checking to avoid division by 0
    if 0. in col_norms:
        raise Exception("Error: One or more columns of the provided matrix has a norm of zero.")

    # Divide column-wise by the L2 norm
    return torch.div(S, col_norms)

def ELBO_simulations(
    D: Integer, 
    N_init: Integer,
    n_actions: Integer,
    truenoise: Float[Tensor, "1 1"],
    n_simulations: Integer,
    n_epochs: Integer,
) -> dict:
    
    simulation_dict = {
        "Method": [], 
        "Simulation": [], 
        "Epoch": [], 
        "N":[],
        "Actions": [],
        "TrueNoise": [],
        "LengthScale": [],
        "OutputScale": [],
        "SigmaSq": [],
        "obsBest": [],
        "trueBest": [],
        "cpuTime": []
    }

    for sim in range(n_simulations):

        # Simulate dataset 
        X = torch.rand(N_init, D)
        y = observe(hartmann_six, X, truenoise)
        true_y = hartmann_six(X)

        # Record initial info for the simulation
        simulation_dict["Method"].append("Separate")
        simulation_dict["Simulation"].append(sim + 1)
        simulation_dict["Epoch"].append(0)
        simulation_dict["N"].append(X.shape[-2])
        simulation_dict["Actions"].append(n_actions)
        simulation_dict["TrueNoise"].append(truenoise.item())
        simulation_dict["LengthScale"].append(np.nan)
        simulation_dict["OutputScale"].append(np.nan)
        simulation_dict["SigmaSq"].append(np.nan)
        
        y_best = y.max().item()
        simulation_dict["obsBest"].append(y_best)
    
        true_best = true_y[y.argmax()].item()
        simulation_dict["trueBest"].append(true_best)

        simulation_dict["cpuTime"].append(np.nan)
        
        # for epoch in tqdm(range(n_epochs), leave = False):
        for epoch in range(n_epochs):
            epoch_st = time.process_time()

            # Initialize S, x_new, GP hyperparameters randomly
            S_raw = torch.randn(X.shape[-2], n_actions)
            ls_raw = torch.randn(1, D)
            os_raw = torch.randn(1, 1)
            sigma_sq_raw = torch.randn(1, 1)
            x_new_raw = torch.rand(1, D).logit()
            

            S_opt = torch.nn.Parameter(S_raw)
            ls_opt = torch.nn.Parameter(ls_raw)
            os_opt = torch.nn.Parameter(os_raw)
            sigma_sq_opt = torch.nn.Parameter(sigma_sq_raw)
            x_new_opt = torch.nn.Parameter(x_new_raw)

            # Optimize S and GP hyperparameters 
            optimizer_S = torch.optim.Adam(params = [S_opt, ls_opt, os_opt, sigma_sq_opt], lr = 0.005, maximize = True)
            for _ in range(500):
                S_normed = normalize_cols(S_opt)
                ls = torch.nn.functional.softplus(ls_opt)
                os = torch.nn.functional.softplus(os_opt)
                sigma_sq = torch.nn.functional.softplus(sigma_sq_opt)

                KXX = matern_kernel(X, X, ls, os) + (sigma_sq + 1e-4) * torch.eye(X.shape[-2]) # epsilon = 1e-4
                STKS = S_normed.mT @ KXX @ S_normed
                try:
                    STKS_chol = torch.linalg.cholesky(STKS)
                except:
                    try:
                        STKS_chol = torch.linalg.cholesky(STKS + 1e-6 * torch.eye(n_actions))
                    except:
                        break   
                        
                gain_S = ELBO(S_normed, X, y, KXX, STKS_chol, ls, os, sigma_sq)
                gain_S.backward()
                optimizer_S.step()
                optimizer_S.zero_grad()
                
            S_normed = normalize_cols(S_opt).detach()          
            STKS = S_normed.mT @ KXX @ S_normed
            try:
                STKS_chol = torch.linalg.cholesky(STKS)
            except:
                try:
                    STKS_chol = torch.linalg.cholesky(STKS + 1e-6 * torch.eye(n_actions))
                except:
                    break 
            STKS_chol.detach_()
            ls.detach_()
            os.detach_()
            sigma_sq.detach_()

            optimizer_x = torch.optim.Adam(params = [x_new_opt], lr = 0.01, maximize = True)
            for _ in range(500):
                # Ensure S, x are "valid"
                x_normed = x_new_opt.sigmoid()  
                
                # Compute the variational inference distribution q_S(f) = f|(S'D) at x_normed
                VI_mean, VI_var = compute_posterior_mean_and_variance_action(x_normed, X, y, S_normed, STKS_chol, ls, os)
                VI_sd = VI_var.clamp(min = 1.0e-10).sqrt()

                # Explicitly compute log(expected improvement)
                z_score = (VI_mean - y_best).div(VI_sd)
                gain_x = VI_sd * (STD_normal.log_prob(z_score).exp() + z_score * STD_normal.cdf(z_score))
                gain_x.backward()
                optimizer_x.step()
                optimizer_x.zero_grad()
    
            # Update data  
            x_new = x_new_opt.sigmoid()       
            x_new.detach_()
            # K_chol = update_chol_newdata(K_chol, X, x_new, ls, os, sigma_sq)
            y_new = observe(hartmann_six, x_new, truenoise)
            true_y_new = hartmann_six(x_new)
            X = torch.cat([X, x_new], -2)
            y = torch.cat([y, y_new], -1)
            true_y = torch.cat([true_y, true_y_new], -1)

            epoch_et = time.process_time()
        
            # Record info for the epoch
            simulation_dict["Method"].append("Separate")
            simulation_dict["Simulation"].append(sim + 1)
            simulation_dict["Epoch"].append(epoch + 1)
            simulation_dict["N"].append(X.shape[-2])
            simulation_dict["Actions"].append(n_actions)
            simulation_dict["TrueNoise"].append(truenoise.item())
            simulation_dict["LengthScale"].append(ls.tolist())
            simulation_dict["OutputScale"].append(os.item())
            simulation_dict["SigmaSq"].append(sigma_sq.item())
        
            y_best = y.max().item()
            simulation_dict["obsBest"].append(y_best)
    
            true_best = true_y[y.argmax()].item()
            simulation_dict["trueBest"].append(true_best)

            simulation_dict["cpuTime"].append(epoch_et - epoch_st)

        # Print progress update
        print(f"Completed {n_epochs} epochs for simulation {sim + 1}!")

    return simulation_dict

def ELBO_sqrt_simulate(
    D: Integer, 
    N_init: Integer,
    truenoise: Float[Tensor, "1 1"],
    n_simulations: Integer,
    n_epochs: Integer,
) -> dict:
    
    simulation_dict = {
        "Method": [], 
        "Simulation": [], 
        "Epoch": [], 
        "N":[],
        "Actions": [],
        "TrueNoise": [],
        "LengthScale": [],
        "OutputScale": [],
        "SigmaSq": [],
        "obsBest": [],
        "trueBest": [],
        "cpuTime": []
    }

    for sim in range(n_simulations):

        # Simulate dataset 
        X = torch.rand(N_init, D)
        y = observe(hartmann_six, X, truenoise)
        true_y = hartmann_six(X)

        # Record initial info for the simulation
        simulation_dict["Method"].append("Separate")
        simulation_dict["Simulation"].append(sim + 1)
        simulation_dict["Epoch"].append(0)
        simulation_dict["N"].append(X.shape[-2])

        n_actions = math.floor(math.sqrt(X.shape[-2]))
        simulation_dict["Actions"].append(n_actions)

        simulation_dict["TrueNoise"].append(truenoise.item())
        simulation_dict["LengthScale"].append(np.nan)
        simulation_dict["OutputScale"].append(np.nan)
        simulation_dict["SigmaSq"].append(np.nan)
        
        y_best = y.max().item()
        simulation_dict["obsBest"].append(y_best)
    
        true_best = true_y[y.argmax()].item()
        simulation_dict["trueBest"].append(true_best)

        simulation_dict["cpuTime"].append(np.nan)
        
        # for epoch in tqdm(range(n_epochs), leave = False):
        for epoch in range(n_epochs):
            epoch_st = time.process_time()

            # Initialize S, x_new, GP hyperparameters randomly
            S_raw = torch.randn(X.shape[-2], n_actions)
            ls_raw = torch.randn(1, D)
            os_raw = torch.randn(1, 1)
            sigma_sq_raw = torch.randn(1, 1)
            x_new_raw = torch.rand(1, D).logit()
            

            S_opt = torch.nn.Parameter(S_raw)
            ls_opt = torch.nn.Parameter(ls_raw)
            os_opt = torch.nn.Parameter(os_raw)
            sigma_sq_opt = torch.nn.Parameter(sigma_sq_raw)
            x_new_opt = torch.nn.Parameter(x_new_raw)

            # Optimize S and GP hyperparameters 
            optimizer_S = torch.optim.Adam(params = [S_opt, ls_opt, os_opt, sigma_sq_opt], lr = 0.005, maximize = True)
            for _ in range(500):
                S_normed = normalize_cols(S_opt)
                ls = torch.nn.functional.softplus(ls_opt)
                os = torch.nn.functional.softplus(os_opt)
                sigma_sq = torch.nn.functional.softplus(sigma_sq_opt)

                KXX = matern_kernel(X, X, ls, os) + (sigma_sq + 1e-4) * torch.eye(X.shape[-2]) # epsilon = 1e-4
                STKS = S_normed.mT @ KXX @ S_normed
                try:
                    STKS_chol = torch.linalg.cholesky(STKS)
                except:
                    try:
                        STKS_chol = torch.linalg.cholesky(STKS + 1e-6 * torch.eye(n_actions))
                    except:
                        break   
                        
                gain_S = ELBO(S_normed, X, y, KXX, STKS_chol, ls, os, sigma_sq)
                gain_S.backward()
                optimizer_S.step()
                optimizer_S.zero_grad()
                
            S_normed = normalize_cols(S_opt).detach()          
            STKS = S_normed.mT @ KXX @ S_normed
            try:
                STKS_chol = torch.linalg.cholesky(STKS)
            except:
                try:
                    STKS_chol = torch.linalg.cholesky(STKS + 1e-6 * torch.eye(n_actions))
                except:
                    break 
            STKS_chol.detach_()
            ls.detach_()
            os.detach_()
            sigma_sq.detach_()

            optimizer_x = torch.optim.Adam(params = [x_new_opt], lr = 0.01, maximize = True)
            for _ in range(500):
                # Ensure S, x are "valid"
                x_normed = x_new_opt.sigmoid()  
                
                # Compute the variational inference distribution q_S(f) = f|(S'D) at x_normed
                VI_mean, VI_var = compute_posterior_mean_and_variance_action(x_normed, X, y, S_normed, STKS_chol, ls, os)
                VI_sd = VI_var.clamp(min = 1.0e-10).sqrt()

                # Explicitly compute log(expected improvement)
                z_score = (VI_mean - y_best).div(VI_sd)
                gain_x = VI_sd * (STD_normal.log_prob(z_score).exp() + z_score * STD_normal.cdf(z_score))
                gain_x.backward()
                optimizer_x.step()
                optimizer_x.zero_grad()
    
            # Update data  
            x_new = x_new_opt.sigmoid()       
            x_new.detach_()
            y_new = observe(hartmann_six, x_new, truenoise)
            true_y_new = hartmann_six(x_new)
            X = torch.cat([X, x_new], -2)
            y = torch.cat([y, y_new], -1)
            true_y = torch.cat([true_y, true_y_new], -1)

            epoch_et = time.process_time()
        
            # Record info for the epoch
            simulation_dict["Method"].append("Separate")
            simulation_dict["Simulation"].append(sim + 1)
            simulation_dict["Epoch"].append(epoch + 1)
            simulation_dict["N"].append(X.shape[-2])

            n_actions = math.floor(math.sqrt(X.shape[-2]))
            simulation_dict["Actions"].append(n_actions)

            simulation_dict["TrueNoise"].append(truenoise.item())
            simulation_dict["LengthScale"].append(ls.tolist())
            simulation_dict["OutputScale"].append(os.item())
            simulation_dict["SigmaSq"].append(sigma_sq.item())
        
            y_best = y.max().item()
            simulation_dict["obsBest"].append(y_best)
    
            true_best = true_y[y.argmax()].item()
            simulation_dict["trueBest"].append(true_best)

            simulation_dict["cpuTime"].append(epoch_et - epoch_st)

        # Print progress update
        print(f"Completed {n_epochs} epochs for simulation {sim + 1}!")

    return simulation_dict

def ELBO_Fire( 
    N_init: Integer,
    n_actions: Integer,
    n_simulations: Integer,
    n_epochs: Integer,
) -> None:
    # Set two constants
    D = 6
    truenoise = torch.zeros(1,1)

    simulation_dict = ELBO_simulations(D, N_init, n_actions, truenoise, n_simulations, n_epochs)
    sim_df = pd.DataFrame(simulation_dict)
        
    # Count existing number of csv files
    n_csvs = len(fnmatch.filter(oper.listdir('./Code/EULBO/Sim-Results/RawData/'), '*.csv'))

    # Save file locally
    sim_df.to_csv(f"./Code/EULBO/Sim-Results/RawData/ELBO_Simulation_Results_{n_csvs}.csv", index = False)

    return None

def ELBO_SQRT( 
    N_init: Integer,
    n_simulations: Integer,
    n_epochs: Integer,
) -> None:
    # Set two constants
    D = 6
    truenoise = torch.zeros(1,1)

    simulation_dict = ELBO_sqrt_simulate(D, N_init, truenoise, n_simulations, n_epochs)
    sim_df = pd.DataFrame(simulation_dict)
        
    # Count existing number of csv files
    n_csvs = len(fnmatch.filter(oper.listdir('./Code/EULBO/Sim-Results/RawData/'), '*.csv'))

    # Save file locally
    sim_df.to_csv(f"./Code/EULBO/Sim-Results/RawData/ELBO_SQRT_Results_{n_csvs}.csv", index = False)

    return None

if __name__ == '__main__':
    fire.Fire({'ELBO_Fire': ELBO_Fire,
               'ELBO_SQRT': ELBO_SQRT})
