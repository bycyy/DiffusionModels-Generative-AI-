import torch
import numpy as np
from functools import partial
import ode_condicional as odec

device = 'cpu'

D = 10
class ExactScoreNet(torch.nn.Module):
    def forward(self, x, t, y):
        # exact score for N(0, \sigma_0^2 I) where marg is N(0, \sigma_t^2 I)
        # for VP SDE, x(t) = x_0 * e^{-0.5 \beta t} + N(0, 1 - e^{-\beta t})
        # If x_0 ~ N(0, 1), then x(t) ~ N(0, 1) forever, score is -x
        return -x 

score_model = ExactScoreNet()

# mock image from N(0, I)
x = torch.randn(4, D)
y = torch.tensor([0, 0, 0, 0])

def bm_drift_coefficient(x, t, y): return -0.5 * x
def bm_diffusion_coefficient(t, y): return torch.ones_like(t) * 1.0
def bm_sigma_t(t, y): return torch.ones_like(t) * 1.0  # if x_0 ~ N(0,1), sigma_t is always 1

bpd_val = odec.compute_bpd_with_ode(
    model=score_model,
    x=x,
    drift_coefficient=bm_drift_coefficient,
    diffusion_coefficient=bm_diffusion_coefficient,
    sigma_t_func=bm_sigma_t,
    y=y,
    n_steps=100,
    seed=42
)

print(bpd_val)
