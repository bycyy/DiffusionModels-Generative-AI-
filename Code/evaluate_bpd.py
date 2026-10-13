import torch
import numpy as np
from functools import partial
import ode_condicional as odec

device = 'cpu'

# mock score model
class MockScoreNet(torch.nn.Module):
    def forward(self, x, t, y):
        # A large neg score means it pushes x towards origin very strongly
        return -100.0 * x

score_model = MockScoreNet()

# mock image
x = torch.rand(4, 3, 28, 28)
y = torch.tensor([0, 1, 2, 3])

def bm_drift_coefficient(x, t, y): return -0.5 * x
def bm_diffusion_coefficient(t, y): return torch.ones_like(t) * 1.0
def bm_sigma_t(t, y): return torch.sqrt(1.0 - torch.exp(-t))

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
