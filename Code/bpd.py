"""
Simulate BPD (bits per dimension) algorithm
"""
# Load packages

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch 
from torch import Tensor
from ode import compute_ODE

def bpd(model, 
        x:Tensor, 
        n_steps:int,
        drift_coefficient: Callable[float, float],
        diffusion_coefficient: Callable[float],
        seed: Union[int, None] = None):
    """Compute BPD of a image x using the model and data process

    Args:
        model: neural network  
        x: image (Tensor)  
        n_steps: number of steps to compute ODE 
        drift_coefficient: Function of :math`(x(t), t)` that defines the drift term            
        diffusion_coefficient: Function of :math`(t)` that defines the diffusion term  
        seed: Seed for the random number generator
    """
    _, _, log_p = compute_ODE(  #Cálculo de NLL, usamos compute ODE
        model=model,
        x_0=x,
        t_0=0.0,
        t_end=1.0,
        n_steps=n_steps,
        drift_coefficient=drift_coefficient,
        diffusion_coefficient=diffusion_coefficient,
        sigma_t=1.0,
        seed=seed
    )

    D = x.shape[1] * x.shape[2] * x.shape[3]

    bpd = - log_p / (D*np.log(2))

    return bpd