"""
Simulate PF-ODE algorithm
"""
# Load packages

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch 
from torch import Tensor

def pf_ode (model, 
            x_T:Tensor, 
            t_0: float,
            t_end: float,
            n_steps:int, 
            drift_coefficient: Callable[float, float],
            diffusion_coefficient: Callable[float],
            seed: Union[int, None] = None
        ):

    """Generation of images using Probability Flow ODE

    Args:
        model: neural network that estimates the score ∇ log p_t(x_t)
        x_T: Original image (pure noise) in t = 1.0
        n_steps: number of integration steps

    Returns:
        x_0: generated image in t = 0.0
    """
    dt = (t_end-t_0)/n_steps  # Δt negativo (SDE reversed, integramos hacia atrás)
    x = x_T.clone()
    device = x_T.device

    times = torch.linspace(t_0, t_end, n_steps+1, device = device)
    
    #set the array that we are going to aproximate to the distribution    
    x_t = torch.tensor(
        np.empty((*np.shape(x_T), len(times))), 
        dtype=torch.float32,
        device=device,
    )
    x_t[..., 0] = x_T

    for i,t in enumerate(times[:-1]):
        t_tensor = torch.full((x.shape[0],), t, device = device)
        score = model(x,t_tensor) #∇ log p_t(x_t)
        dx_dt = drift_coefficient(x,t_tensor) - 0.5*diffusion_coefficient(t_tensor)**2 * score #dx/dt = f(x,t) - ½·g(t)²·∇ log p_t(x_t)

        x = x + dx_dt * dt
        x_t[..., i + 1] = x
    return times, x_t

def compute_bpd_with_ode(model, x, drift_coefficient, 
                        diffusion_coefficient, sigma_t_func, n_steps=100, seed=None):
    """
    Calcula BPD usando la ODE de Probability Flow
    """
    device = x.device
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Integrar ODE de eps=1e-3 a t=1
    eps_time = 1e-3
    t_end = 1.0
    dt = (t_end - eps_time) / n_steps
    x_t = x.clone()
    log_p = torch.zeros(x.shape[0], device=device)
    
    for i in range(n_steps):
        t = eps_time + i * dt
        t_tensor = torch.full((x.shape[0],), t, device=device)
        
        with torch.enable_grad():
            x_t.requires_grad_(True)
            # ODE: dx/dt = f(x,t) - 0.5 * g(t)^2 * score
            score = model(x_t, t_tensor)
            drift = drift_coefficient(x_t, t_tensor)
            diffusion = diffusion_coefficient(t_tensor)
            
            v = drift - 0.5 * (diffusion**2) * score
            
            # Calcular divergencia
            eps = torch.randn_like(x_t)
            grad_v_eps = torch.autograd.grad(v, x_t, grad_outputs=eps, create_graph=False)[0]
            div_v = torch.sum(grad_v_eps * eps, dim=(1, 2, 3))
            
            log_p = log_p + div_v * dt
        
        x_t = x_t + v * dt
        x_t = x_t.detach()
    
    # Prior
    D = x[0].numel()
    sigma_T = sigma_t_func(torch.tensor(1.0, device=device))
    
    log_p_t = -torch.norm(x_t.view(x_t.shape[0], -1), dim=1)**2 / (2 * sigma_T**2)
    log_p_t = log_p_t - (D / 2) * torch.log(2 * torch.pi * sigma_T**2)
    
    log_likelihood = log_p + log_p_t
    bpd = -log_likelihood / (D * np.log(2))
    
    # Ajuste de "dequantization" para imágenes [0, 1] representadas con 256 valores (8 bits por píxel)
    bpd = bpd + 8.0
    
    return bpd