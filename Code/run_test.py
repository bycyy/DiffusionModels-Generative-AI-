import torch
from functools import partial
import ode_condicional as odec
from score_model import ScoreNet
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor

device = 'cpu'
T = 1.0
beta_min = 0.1
beta_max = 20.0

def vp_sigma_t_unbatched(t):
    integral_beta = beta_min * t + 0.5 * (beta_max - beta_min) * ((t**2)/T)
    # matching the 1e-8 term that avoids NaN
    return torch.sqrt(1.0 - torch.exp(-integral_beta) + 1e-8)

score_model = ScoreNet(
    marginal_prob_std=vp_sigma_t_unbatched, 
    num_classes=10,
    input_channels=3
).to(device)

def remove_module_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

try:
    sd = torch.load('check_point.pth', map_location=device)
    score_model.load_state_dict(remove_module_prefix(sd))
    score_model.eval()
except Exception as e:
    print("Error loading point:", e)

def vp_drift_coefficient(x_t, t, y):
    t_tensor = t.view(-1, 1, 1, 1)
    beta = beta_min + t_tensor * (beta_max - beta_min)
    return -0.5 * beta * x_t

def vp_diffusion_coefficient(t, y):
    beta = beta_min + t * (beta_max - beta_min)
    return torch.sqrt(beta)

def vp_sigma_t(t, y):
    t = torch.tensor(t, device=device)
    integral_beta = beta_min * t + 0.5 * (beta_max - beta_min) * ((t**2)/T)
    return torch.sqrt(1.0 - torch.exp(-integral_beta) + 1e-8)

print("Loading some data...")
data = datasets.MNIST(root='data', train=True, download=True, transform=ToTensor())
# convert to 3 channels since model expects 3
x = torch.stack([data[i][0] for i in range(4)]).to(device)
x = x.repeat(1, 3, 1, 1)
y = torch.tensor([data[i][1] for i in range(4)]).to(device)

def custom_bpd(model, x, drift_coefficient, diffusion_coefficient, sigma_t_func, y, n_steps=100, seed=42):
    device = x.device
    if seed is not None:
        torch.manual_seed(seed)
    
    epsilon = 1e-3
    T_end = 1.0
    dt = (T_end - epsilon) / n_steps
    x_t = x.clone()
    log_p = torch.zeros(x.shape[0], device=device)
    
    for i in range(n_steps):
        t = epsilon + i * dt
        t_tensor = torch.full((x.shape[0],), t, device=device)
        
        with torch.enable_grad():
            x_t.requires_grad_(True)
            score = model(x_t, t_tensor, y)
            drift = drift_coefficient(x_t, t_tensor, y)
            diffusion = diffusion_coefficient(t_tensor, y)
            if torch.is_tensor(diffusion) and diffusion.ndim > 0 and diffusion.shape[0] == x.shape[0]:
                diffusion = diffusion.view(-1, *([1] * (x_t.ndim - 1)))
                
            v = drift - 0.5 * (diffusion**2) * score
            eps = torch.randn_like(x_t)
            grad_v_eps = torch.autograd.grad(v, x_t, grad_outputs=eps, create_graph=False)[0]
            div_v = torch.sum(grad_v_eps * eps, dim=(1, 2, 3))
            
            log_p = log_p + div_v * dt
        
        x_t = x_t + v * dt
        x_t = x_t.detach()
    
    D = x[0].numel()
    sigma_T = sigma_t_func(torch.tensor(1.0, device=device), y)
    if torch.is_tensor(sigma_T) and sigma_T.ndim > 0:
        sigma_T = sigma_T.view(-1)
        log_p_t = -torch.norm(x_t.view(x_t.shape[0], -1), dim=1)**2 / (2 * sigma_T**2)
        log_p_t = log_p_t - (D / 2) * torch.log(2 * torch.pi * sigma_T**2)
    else:
        log_p_t = -torch.norm(x_t.view(x_t.shape[0], -1), dim=1)**2 / (2 * sigma_T**2)
        log_p_t = log_p_t - (D / 2) * torch.log(2 * torch.pi * sigma_T**2)
    
    log_likelihood = log_p + log_p_t
    bpd = -log_likelihood / (D * np.log(2))
    bpd = bpd + 8.0
    return log_p, log_p_t, bpd

print("evaluating bpd...")
log_p, log_p_t, bpd_values = custom_bpd(
    model=score_model,
    x=x,
    drift_coefficient=vp_drift_coefficient,
    diffusion_coefficient=vp_diffusion_coefficient,
    sigma_t_func=vp_sigma_t,
    y=y,
    n_steps=100,
    seed=42
)
print("log_p:", log_p.mean().item())
print("log_p_t:", log_p_t.mean().item())
print("BPD values:", bpd_values)
