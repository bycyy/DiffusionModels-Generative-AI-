# -*- coding: utf-8 -*-
"""
Simulate Gaussian processes.

@author: <alberto.suarez@uam.es>
"""
# Load packages

from __future__ import annotations

from typing import Callable, Union, Any

import numpy as np
import torch
from torch import Tensor
import inspect

def _safe_call(func: Callable, *args) -> Tensor:
    """Helper to call a function with the correct number of arguments."""
    try:
        return func(*args)
    except TypeError as e:
        # Check if the error is likely due to argument count mismatch
        sig = inspect.signature(func)
        arg_count = len(sig.parameters)
        
        if arg_count < len(args):
            return func(*args[:arg_count])
        raise e

def euler_maruyama_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,  
    drift_coefficient: Callable[[Tensor, Tensor, Tensor], Tensor],
    diffusion_coefficient: Callable[[Tensor, Tensor], Tensor],
    y: Tensor, 
    model: Any = None,
    seed: Union[int, None] = None,
    clip: bool = False
) -> Tensor:
    """Euler-Maruyama integrator (approximate)

     Args:
        x_0: The initial images of dimensions 
            (batch_size, n_channels, image_height, image_width)
        t_0: float,
        t_end: endpoint of the integration interval    
        n_steps: number of integration steps 
        drift_coefficient: Function of :math`(x(t), t, y)` that defines the drift term            
        diffusion_coefficient: Function of :math`(t, y)` that defines the diffusion term  
        y: Labels for the images (batch_size,)
        seed: Seed for the random number generator
        
    Returns:
        x_t: Trajectories that result from the integration of the SDE.
             The shape is (*np.shape(x_0), (n_steps + 1))
            
    Notes:
        The implementation is fully vectorized except for a loop over time.

    Examples:
        >>> import numpy as np
        >>> drift_coefficient = lambda x_t, t, y: - x_t
        >>> diffusion_coefficient = lambda t, y: torch.ones_like(t)
        >>> x_0 = torch.tensor(np.reshape(np.arange(120), (2, 3, 5, 4)))
        >>> y = torch.zeros(2, dtype=torch.long)
        >>> t_0, t_end = 0.0, 3.0
        >>> n_steps = 6
        >>> times, x_t = euler_maruyama_integrator(
        ...     x_0, t_0, t_end, n_steps, drift_coefficient, diffusion_coefficient, y
        ... )
        >>> print(times)
        tensor([0.0000, 0.5000, 1.0000, 1.5000, 2.0000, 2.5000, 3.0000])
        >>> print(np.shape(x_t))
        torch.Size([2, 3, 5, 4, 7])
    """
    
    # [TO DO: Include short comments to describe operation of the code]
    #Sets the device to the same of the vector x_0
    device = x_0.device
   
    #sets the time array to the interval t_0, t_end, and sets the incremental time
    times = torch.linspace(t_0, t_end, n_steps + 1, device=device)
    dt = times[1] - times[0]  

    #set the array that we are going to aproximate to the distribution    
    x_t = torch.empty((*x_0.shape, len(times)), 
        dtype=torch.float32,
        device=device,
    )
    x_t[..., 0] = x_0
    
    #generates the random noise
    z = torch.randn_like(x_t)
    z[..., -1] = 0.0 # No noise injection in the last step
    
    # Main loop for Euler-Maruyama integration
    for n, t in enumerate(times[:-1]): 
        # Create a batch of time values
        t_batch = torch.full((x_0.shape[0],), t, device=device)
        
        # Calculate the next state using the drift and diffusion coefficients
        drift = _safe_call(drift_coefficient, x_t[..., n], t_batch, y)
        diffusion = _safe_call(diffusion_coefficient, t_batch, y)
        
        # Reshape for broadcasting if necessary
        # diffusion is expected to be (batch_size,)
        if torch.is_tensor(diffusion) and diffusion.ndim > 0 and diffusion.shape[0] == x_0.shape[0]:
            diffusion = diffusion.view(-1, *([1] * (x_0.ndim - 1)))
            
        if model is not None:
            # If a score model is provided, we assume we are simulating the reverse SDE
            # for diffusion sampling. The drift is modified by the score:
            # f_rev = f_fwd - g^2 * score
            score = model(x_t[..., n], t_batch, y)
            drift = drift - (diffusion**2) * score

        x_t[..., n + 1] = ( 
            x_t[..., n]
            + drift * dt
            + diffusion
            * torch.sqrt(torch.abs(dt))
            * z[..., n]
        )
        
    if clip:
        final = x_t[..., -1]            # shape: (batch, C, H, W)
        B = final.shape[0]
        flat = final.view(B, -1)        # (batch, C*H*W)

        x_t[..., -1] = final.clamp(0.0, 1.0)
        
    return times, x_t
    
class DiffussionProcess:

    def __init__(
        self,          
        drift_coefficient: Callable[[Tensor, Tensor, Tensor], Tensor] = lambda x_t, t, y: 0.0,
        diffusion_coefficient: Callable[[Tensor, Tensor], Tensor] = lambda t, y: 1.0,
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
     

        
class GaussianDiffussionProcess(DiffussionProcess):
    """
    
    [TO DO: Complete the docstring]
    
    Example 1:
        >>> mu, sigma = 1.5, 2.0
        >>> bm = GaussianDiffussionProcess(
        ...     drift_coefficient=lambda x_t, t, y: mu,
        ...     diffusion_coefficient=lambda t, y: sigma,
        ...     mu_t=lambda x_0, t, y: x_0 + mu*t,
        ...     sigma_t=lambda t, y: torch.sqrt(2.0 * t),
        ... )
        >>> print(bm.drift_coefficient(3.0, 10.0, None))
        1.5
        >>> print(bm.diffusion_coefficient(10.0, None))
        2.0
        >>> print(bm.mu_t(3.0, 10.0, None), bm.sigma_t(torch.tensor(10.0), None))
        18.0 tensor(4.4721)
        

    """
    kind = "Gaussian"
    
    def __init__(
        self,          
        drift_coefficient: Callable[[Tensor, Tensor, Tensor], Tensor] = lambda x_t, t, y: 0.0,
        diffusion_coefficient: Callable[[Tensor, Tensor], Tensor] = lambda t, y: 1.0,
        mu_t: Callable[[Tensor, Tensor, Tensor], Tensor] = lambda x_0, t, y: x_0,
        sigma_t: Callable[[Tensor, Tensor], Tensor] = lambda t, y: torch.sqrt(t),
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
        self.mu_t = mu_t
        self.sigma_t = sigma_t
    

    def loss_function(
        self,
        score_model, 
        x_0: torch.Tensor, 
        y: torch.Tensor,
        eps: float = 1.0e-5,
    ):
        """The loss function for training score-based generative models.

          Args:
              score_model:  A PyTorch model instance that represents a 
                            time-dependent score-based model.
          x_0: A mini-batch of training data.    
          y: A mini-batch of training labels.
          eps: A tolerance value for numerical stability.
        """
        
        # 1. Muestrear t para TODO el batch simultáneamente
        # t tendrá forma (batch_size,)
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps  
        
        # 2. Generar ruido blanco para todo el batch
        z = torch.randn_like(x_0)
        
        # 3. Obtener sigma_t y reajustar dimensiones para broadcasting
        # sigmas pasará de (batch_size,) a (batch_size, 1, 1, 1)
        sigmas = _safe_call(self.sigma_t, t, y)
        if not torch.is_tensor(sigmas):
            sigmas = torch.tensor(sigmas, device=x_0.device, dtype=x_0.dtype)
        
        # Ensure sigmas is at least 1D for broadcasting
        # If it's a scalar or 1-element tensor, expand it to the batch size
        if sigmas.ndim == 0 or (sigmas.ndim == 1 and sigmas.shape[0] == 1):
            sigmas = sigmas.expand(x_0.shape[0])
            
        sigmas_reshaped = sigmas.view(-1, *([1] * (x_0.ndim - 1)))
        
        # 4. Perturbar la imagen (En VE, mu_t es simplemente x_0)
        # x_t = x_0 + sigma_t * z
        #x_t = x_0 + sigmas_reshaped * z
        mu = _safe_call(self.mu_t, x_0, t, y)
        x_t = mu + sigmas_reshaped * z
        
        # 5. Estimar el score con el modelo (passing the labels y)
        score = score_model(x_t, t, y)
        
        # 6. Calcular la pérdida (Denoising Score Matching)
        # La fórmula es: E[ || score * sigma + z ||^2 ]
        # El resultado de 'loss' será un tensor escalar (0-d tensor)
        losses = (score * sigmas_reshaped + z) ** 2
        loss = torch.mean(losses)
        
        return loss


if __name__ == "__main__":
    import doctest
    doctest.testmod()
