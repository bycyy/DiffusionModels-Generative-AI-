# -*- coding: utf-8 -*-
"""
Simulate Gaussian processes.

@author: <alberto.suarez@uam.es>
"""
# Load packages

from __future__ import annotations

from typing import Callable, Union

import numpy as np
import torch
from torch import Tensor

def euler_maruyama_integrator(
    x_0: Tensor,
    t_0: float,
    t_end: float,
    n_steps: int,  
    drift_coefficient: Callable[float, float],
    diffusion_coefficient: Callable[float],
    seed: Union[int, None] = None
) -> Tensor:
    """Euler-Maruyama integrator (approximate)

     Args:
        x_0: The initial images of dimensions 
            (batch_size, n_channels, image_height, image_width)
        t_0: float,
        t_end: endpoint of the integration interval    
        n_steps: number of integration steps 
        drift_coefficient: Function of :math`(x(t), t)` that defines the drift term            
        diffusion_coefficient: Function of :math`(t)` that defines the diffusion term  
        seed: Seed for the random number generator
        
    Returns:
        x_t: Trajectories that result from the integration of the SDE.
             The shape is (*np.shape(x_0), (n_steps + 1))
            
    Notes:
        The implementation is fully vectorized except for a loop over time.

    Examples:
        >>> import numpy as np
        >>> drift_coefficient = lambda x_t, t: - x_t
        >>> diffusion_coefficient = lambda t: torch.ones_like(t)
        >>> x_0 = torch.tensor(np.reshape(np.arange(120), (2, 3, 5, 4)))
        >>> t_0, t_end = 0.0, 3.0
        >>> n_steps = 6
        >>> times, x_t = euler_maruyama_integrator(
        ...     x_0, t_0, t_end, n_steps, drift_coefficient, diffusion_coefficient, 
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
    x_t = torch.tensor(
        np.empty((*np.shape(x_0), len(times))), 
        dtype=torch.float32,
        device=device,
    )
    x_t[..., 0] = x_0
    
    #generates the random noise
    z = torch.randn_like(x_t)
    z[..., -1] = 0.0 # No noise injection in the last step
    
    #euler_matuyama process
    for n, t in enumerate(times[:-1]): 
        t = torch.ones(x_0.shape[0], device=device) * t
        #Updates the values of the distribution for getting to the selected
        x_t[..., n + 1] = ( 
            x_t[..., n]
            + drift_coefficient(x_t[..., n], t) * dt
            + diffusion_coefficient(t)
            * torch.sqrt(torch.abs(torch.tensor(dt, device=x_t.device)))
            * z[..., n]
        )
        
    return times, x_t
    
class DiffussionProcess:

    def __init__(
        self,          
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
     

        
class GaussianDiffussionProcess(DiffussionProcess):
    """
    Gaussian diffussion process used in score-based generative models

    Args:
        drift_coefficient: Callable function that defines de drift term
        diffussion_coefficient: Callable function that defines de diffusion term
        mu_t: Callable function that computes the conditional mean
        sigma_t: Callable function that computes the conditional standard deviation
    
    Example 1:
        >>> mu, sigma = 1.5, 2.0
        >>> bm = GaussianDiffussionProcess(
        ...     drift_coefficient=lambda x_t, t: mu,
        ...     diffusion_coefficient=lambda t: sigma,
        ...     mu_t=lambda x_0, t: x_0 + mu*t,
        ...     sigma_t=lambda t: np.sqrt(2.0 * t),
        ... )
        >>> print(bm.drift_coefficient(x_t=3.0, t=10.0))
        1.5
        >>> print(bm.diffusion_coefficient(t=10.0))
        2.0
        >>> print(bm.mu_t(x_0=3.0, t=10.0), bm.sigma_t(t=10.0))
        18.0 4.47213595499958
        

    """
    kind = "Gaussian"
    
    def __init__(
        self,          
        drift_coefficient: Callable[float, float] = lambda x_t, t: 0.0,
        diffusion_coefficient: Callable[float] = lambda t: 1.0,
        mu_t: Callable[float, float] = lambda x_0, t: x_0,
        sigma_t: Callable[float] = lambda t: np.sqrt(t),
    ):
        self.drift_coefficient = drift_coefficient
        self.diffusion_coefficient = diffusion_coefficient
        self.mu_t = mu_t
        self.sigma_t = sigma_t
    

    def loss_function(
        self,
        score_model, 
        x_0: torch.Tensor, 
        eps: float = 1.0e-5,
    ):
        """The loss function for training score-based generative models.

          Args:
              score_model:  A PyTorch model instance that represents a 
                            time-dependent score-based model.
          x_0: A mini-batch of training data.    
          eps: A tolerance value for numerical stability.
        """
        
        # 1. Muestrear t para TODO el batch simultáneamente
        # t tendrá forma (batch_size,)
        t = torch.rand(x_0.shape[0], device=x_0.device) * (1.0 - eps) + eps  
        
        # 2. Generar ruido blanco para todo el batch
        z = torch.randn_like(x_0)
        
        # 3. Obtener sigma_t y reajustar dimensiones para broadcasting
        # sigmas pasará de (batch_size,) a (batch_size, 1, 1, 1)
        sigmas = self.sigma_t(t)
        sigmas_reshaped = sigmas.view(-1, 1, 1, 1)
        
        # 4. Perturbar la imagen (En VE, mu_t es simplemente x_0)
        # x_t = x_0 + sigma_t * z
        #x_t = x_0 + sigmas_reshaped * z
        x_t = self.mu_t(x_0,t) + sigmas_reshaped * z
        
        # 5. Estimar el score con el modelo
        score = score_model(x_t, t)
        
        # 6. Calcular la pérdida (Denoising Score Matching)
        # La fórmula es: E[ || score * sigma + z ||^2 ]
        # El resultado de 'loss' será un tensor escalar (0-d tensor)
        losses = (score * sigmas_reshaped + z) ** 2
        loss = torch.mean(losses)
        
        return loss


if __name__ == "__main__":
    import doctest
    doctest.testmod()
