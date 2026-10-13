"""
Algorithms for different noise schedules
"""
# Load packages

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch 
from torch import Tensor

def linear_schedule(t,T, beta_min = 0.00001, beta_max=0.02):
    """Linear noise schedule

    Args:
        t: actual step (0-T) 
        T: total number of steps
        beta_min: minimum noise. Defaults to 0.00001.
        beta_max: maximum noise. Defaults to 0.02.
    """
    return beta_min + (t/T) * (beta_max - beta_min)

def cosine_schedule(t, s=0.008):
    """Cosine noise schedule

    Args:
        t: normalized time in [0,1]
        s: offset. Defaults to 0.008 
    """
    f_t = torch.cos((t+s)/(1+s)*torch.pi/2)**2
    f_0 = torch.cos(torch.tensor(s/(1+s)*torch.pi/2))**2
    return f_t / f_0

def square_root_schedule(t,T,s=0.0001):
    """Square root schedule

    Args:
        t: actual step (0-T)
        T: Total number of steps
        s: offset. Defaults to 0.0001
    """

    return 1 - torch.sqrt(t/T+s)

def linear_beta(t, beta_min=0.1, beta_max=20.0):
    """Linear beta schedule for VP-SDE."""
    return beta_min + t * (beta_max - beta_min)

def linear_alpha(t, beta_min=0.1, beta_max=20.0):
    """Cumulative alpha for a linear beta schedule."""
    integral_beta = beta_min * t + 0.5 * (t**2) * (beta_max - beta_min)
    return torch.exp(-integral_beta)

def vp_linear_drift(x_t, t, beta_min=0.1, beta_max=20.0):
    """Drift coefficient for VP-SDE with linear beta schedule."""
    beta = linear_beta(t, beta_min, beta_max)
    if torch.is_tensor(beta) and beta.ndim > 0:
        beta = beta.view(-1, *([1] * (x_t.ndim - 1)))
    return -0.5 * beta * x_t

def vp_linear_diffusion(t, beta_min=0.1, beta_max=20.0):
    """Diffusion coefficient for VP-SDE with linear beta schedule."""
    beta = linear_beta(t, beta_min, beta_max)
    return torch.sqrt(beta + 1e-8)

def vp_linear_mu(x_0, t, beta_min=0.1, beta_max=20.0):
    """Mean of p(x_t|x_0) for VP-SDE with linear beta schedule."""
    alpha = linear_alpha(t, beta_min, beta_max)
    if torch.is_tensor(alpha) and alpha.ndim > 0:
        alpha = alpha.view(-1, *([1] * (x_0.ndim - 1)))
    return x_0 * torch.sqrt(alpha + 1e-8)

def vp_linear_sigma(t, beta_min=0.1, beta_max=20.0):
    """Standard deviation of p(x_t|x_0) for VP-SDE with linear beta schedule."""
    alpha = linear_alpha(t, beta_min, beta_max)
    return torch.sqrt(1.0 - alpha + 1e-8)