"""
Simulate PF-ODE algorithm (conditional version)
"""
# Load packages

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch 
from torch import Tensor
import inspect

def _safe_call(func: Callable, *args) -> Tensor:
    """Helper to call a function with the correct number of arguments."""
    try:
        return func(*args)
    except TypeError:
        # Check the number of arguments the function expects
        sig = inspect.signature(func)
        if hasattr(func, "func"): # Handle partial
             arg_count = len(inspect.signature(func.func).parameters) - len(func.args)
        else:
             arg_count = len(sig.parameters)
        
        if arg_count < len(args):
            return func(*args[:arg_count])
        raise

def _compute_drift_ode(model, x, t_tensor, y, drift_coefficient, diffusion_coefficient):
    """Calcula dx/dt = f(x,t) - ½·g(t)²·∇ log p_t(x_t) para la PF-ODE."""
    score = model(x, t_tensor, y)
    drift = _safe_call(drift_coefficient, x, t_tensor, y)
    diffusion = _safe_call(diffusion_coefficient, t_tensor, y)
    
    # Reshape diffusion for broadcasting if necessary
    if torch.is_tensor(diffusion) and diffusion.ndim > 0 and diffusion.shape[0] == x.shape[0]:
        diffusion = diffusion.view(-1, *([1] * (x.ndim - 1)))
        
    return drift - 0.5 * (diffusion**2) * score, score, diffusion


def pf_ode (model, 
            x_T:Tensor, 
            t_0: float,
            t_end: float,
            n_steps:int, 
            drift_coefficient: Callable[float, float],
            diffusion_coefficient: Callable[float],
            y: Tensor, 
            seed: Union[int, None] = None,
            clip: bool = False,
            method: str = 'heun',
            corrector_steps: int = 0,
            corrector_snr: float = 0.16,
            denoise: bool = True,
            sigma_t_fn: Callable = None,
        ):

    """Generation of images using Probability Flow ODE

    Args:
        model: neural network that estimates the score ∇ log p_t(x_t)
        x_T: Original image (pure noise) in t = 1.0
        n_steps: number of integration steps
        y: labels for the generated images
        method: integration method - 'euler' (1st order) or 'heun' (2nd order, 
                more accurate per step, produces sharper digits)
        corrector_steps: number of Langevin corrector steps per timestep.
                         Each step refines the sample using the score, improving
                         detail fidelity (0 = no correction, 1-2 recommended)
        corrector_snr: signal-to-noise ratio for corrector (controls step size).
                       Lower values = more conservative, less distortion.
        denoise: if True, apply Tweedie denoising at the final timestep
                 to analytically remove residual noise (x̂₀ = x + σ²·∇log p)
        sigma_t_fn: function σ(t) needed for Tweedie denoising and corrector.
                    Required if denoise=True or corrector_steps>0.

    Returns:
        times: timestep values
        x_t: trajectory tensor with shape (*x_T.shape, n_steps+1)
    """
    dt = (t_end-t_0)/n_steps  # Δt negativo (SDE reversed, integramos hacia atrás)
    x = x_T.clone()
    device = x_T.device

    times = torch.linspace(t_0, t_end, n_steps+1, device = device)
    
    #set the array that we are going to aproximate to the distribution    
    x_t = torch.empty((*x_T.shape, len(times)), 
        dtype=torch.float32,
        device=device,
    )
    x_t[..., 0] = x_T

    for i,t in enumerate(times[:-1]):
        t_tensor = torch.full((x.shape[0],), t, device = device)
        
        # ── Predictor-Corrector: Langevin corrector steps ──
        # Refina la muestra x usando pasos de Langevin MCMC en el tiempo t actual.
        # Cada paso toma un gradiente del score y añade ruido calibrado,
        # moviendo x hacia regiones de mayor probabilidad bajo p_t(x).
        if corrector_steps > 0 and sigma_t_fn is not None:
            for _ in range(corrector_steps):
                score_c = model(x, t_tensor, y)
                sigma_c = _safe_call(sigma_t_fn, t_tensor, y)
                if torch.is_tensor(sigma_c) and sigma_c.ndim > 0:
                    sigma_c = sigma_c.view(-1, *([1] * (x.ndim - 1)))
                # Paso de Langevin: tamaño del paso proporcional a σ² y SNR
                noise = torch.randn_like(x)
                grad_norm = torch.norm(score_c.reshape(score_c.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (corrector_snr * noise_norm / (grad_norm + 1e-8)) ** 2 * 2
                x = x + step_size * score_c + torch.sqrt(2 * step_size) * noise
        
        # ── Predictor: integración ODE ──
        dx_dt, score, diffusion = _compute_drift_ode(
            model, x, t_tensor, y, drift_coefficient, diffusion_coefficient)
        
        if method == 'heun' and i < n_steps - 1:
            # Heun's method (Improved Euler / 2nd-order Runge-Kutta):
            # 1) Predicción Euler: x̃ = x + f(x,t)·dt
            x_pred = x + dx_dt * dt
            # 2) Evaluar el drift en el punto predicho (t + dt)
            t_next = torch.full((x.shape[0],), times[i+1], device=device)
            dx_dt_next, _, _ = _compute_drift_ode(
                model, x_pred, t_next, y, drift_coefficient, diffusion_coefficient)
            # 3) Corrección: promedio de ambas pendientes
            x = x + 0.5 * (dx_dt + dx_dt_next) * dt
        else:
            # Euler estándar (1st order)
            x = x + dx_dt * dt
            
        x_t[..., i + 1] = x
    
    # ── Tweedie denoising final ──
    # En t_end aún queda ruido residual σ(t_end). La fórmula de Tweedie
    # permite eliminarlo analíticamente: x̂₀ = x + σ²·∇log p_t(x)
    if denoise and sigma_t_fn is not None:
        t_final = torch.full((x.shape[0],), times[-1], device=device)
        score_final = model(x_t[..., -1], t_final, y)
        sigma_final = _safe_call(sigma_t_fn, t_final, y)
        if torch.is_tensor(sigma_final) and sigma_final.ndim > 0:
            sigma_final = sigma_final.view(-1, *([1] * (x.ndim - 1)))
        x_t[..., -1] = x_t[..., -1] + (sigma_final ** 2) * score_final
        
    if clip:
        x_t[..., -1] = x_t[..., -1].clamp(0.0, 1.0)

    return times, x_t



def compute_ODE(model, 
            x_0:Tensor, 
            t_0: float,
            t_end: float,
            n_steps:int, 
            drift_coefficient: Callable[float, float],
            diffusion_coefficient: Callable[float],
            y: Tensor,
            sigma_t: float =1.0,
            seed: Union[int, None] = None
        )-> Tensor:
    """_summary_

    Args:
        model (_type_): _description_
        x_0 (Tensor): _description_
        t_0 (float): _description_
        t_end (float): _description_
        n_steps (int): _description_
        drift_coefficient (Callable[float, float]): _description_
        diffusion_coefficient (Callable[float]): _description_
        seed (Union[int, None], optional): _description_. Defaults to None.
    """

    device = x_0.device

    if seed is not None:
        torch.manual_seed(seed)

    times = torch.linspace(t_end, t_0, n_steps+1, device = device)
    dt = (t_end-t_0)/n_steps
    x = x_0.clone()
    sigma_t = torch.tensor(sigma_t, device=x_0.device, dtype=x_0.dtype)
    log_p = torch.zeros(x.shape[0], device = device)
    
    #set the array that we are going to aproximate to the distribution    
    x_t = torch.empty((*x_0.shape, len(times)), 
        dtype=torch.float32,
        device=device,
    )
    x_t[..., 0] = x_0

    for i,t in enumerate(times[:-1]): #Es la integral
        t_tensor = torch.full((x.shape[0],), t, device = device)

        with torch.enable_grad():
            x.requires_grad_(True) #Habilitamos cálculo de gradientes
            score = model(x, t_tensor, y) #∇ log p_t(x_t) with labels y
            
            drift = _safe_call(drift_coefficient, x, t_tensor, y)
            diffusion = _safe_call(diffusion_coefficient, t_tensor, y)
            
            # Reshape diffusion for broadcasting if necessary
            if torch.is_tensor(diffusion) and diffusion.ndim > 0 and diffusion.shape[0] == x.shape[0]:
                diffusion = diffusion.view(-1, *([1] * (x.ndim - 1)))
                
            v = drift - 0.5 * (diffusion**2) * score #dx/dt = f(x,t) - ½·g(t)²·∇ log p_t(x_t)

            #Cálculo de divergencia ∇·v usando aproximación de Hutchison (explicación en .ipynb)
            eps = torch.randn_like(x)
            grad_v_eps = torch.autograd.grad(v,x,grad_outputs = eps, create_graph = False)[0]

            div_v = torch.sum(grad_v_eps*eps, dim=(1,2,3))

            log_p = log_p + div_v*dt

        x = x + v * dt #Actualización de x usando Euler
        x = x.detach()

        x_t[..., i + 1] = x

    D = x_0[0].numel() #número de dimensiones por imagen
    x_T = x
    log_p_t =  -torch.norm(x_T.view(x_T.shape[0], -1), dim=1)**2 / (2 * sigma_t**2)
    log_p_t = log_p_t - (D/2)*torch.log(2*torch.pi*sigma_t**2)

    log_likelihood = log_p + log_p_t

    return times, x_t, log_likelihood


def compute_bpd_with_ode(model, 
                        x: Tensor, 
                        drift_coefficient: Callable,
                        diffusion_coefficient: Callable,
                        sigma_t_func: Callable,
                        y: Tensor,
                        n_steps: int = 100, 
                        seed: Union[int, None] = None
                    ) -> Tensor:
    """
    Calcula BPD usando la ODE de Probability Flow (versión condicional)
    """
    device = x.device
    
    if seed is not None:
        torch.manual_seed(seed)
    
    # Integrar ODE de eps=1e-3 a t=1 (t=1 es t_end típicamente)
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
            
            score = model(x_t, t_tensor, y)
            drift = _safe_call(drift_coefficient, x_t, t_tensor, y)
            diffusion = _safe_call(diffusion_coefficient, t_tensor, y)
            
            # Reshape diffusion for broadcasting si es necesario
            if torch.is_tensor(diffusion) and diffusion.ndim > 0 and diffusion.shape[0] == x.shape[0]:
                diffusion = diffusion.view(-1, *([1] * (x_t.ndim - 1)))
                
            # ODE de Probability Flow completo
            v = drift - 0.5 * (diffusion**2) * score
            
            # Calcular divergencia usando Hutchinson
            eps = torch.randn_like(x_t)
            grad_v_eps = torch.autograd.grad(v, x_t, grad_outputs=eps, create_graph=False)[0]
            div_v = torch.sum(grad_v_eps * eps, dim=(1, 2, 3))
            
            log_p = log_p + div_v * dt
        
        x_t = x_t + v * dt
        x_t = x_t.detach()
    
    # Prior at T=1.0
    D = x[0].numel()
    sigma_T = _safe_call(sigma_t_func, torch.tensor(1.0, device=device), y)
    if torch.is_tensor(sigma_T) and sigma_T.ndim > 0:
        sigma_T = sigma_T.view(-1)
        log_p_t = -torch.norm(x_t.view(x_t.shape[0], -1), dim=1)**2 / (2 * sigma_T**2)
        log_p_t = log_p_t - (D / 2) * torch.log(2 * torch.pi * sigma_T**2)
    else:
        log_p_t = -torch.norm(x_t.view(x_t.shape[0], -1), dim=1)**2 / (2 * sigma_T**2)
        log_p_t = log_p_t - (D / 2) * torch.log(2 * torch.pi * sigma_T**2)
    
    log_likelihood = log_p + log_p_t
    bpd = -log_likelihood / (D * np.log(2))
    
    # Ajuste de "dequantization" para imágenes [0, 1] (añadir 8 bits per dimension)
    bpd = bpd + 8.0
    
    return bpd
