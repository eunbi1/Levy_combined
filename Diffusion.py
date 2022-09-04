import torch

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'




class VPSDE:
    def __init__(self, alpha, beta_min=0.1, beta_max=20, T=1., device=device, b=0.9,c=0.1):
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.alpha = alpha
        self.b =b
        self.c = c
        self.T = T

    def beta(self, t):
        return (self.beta_1 - self.beta_0) * t + self.beta_0

    def marginal_log_mean_coeff(self, t):
        log_alpha_t = - 1 / (2 * 2) * (t ** 2) * (self.beta_1 - self.beta_0) - 1 / 2 * t * self.beta_0
        return log_alpha_t

    def diffusion_coeff(self, t):
        return torch.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        sigma1 = self.b*torch.pow(1. - torch.exp(2 * self.marginal_log_mean_coeff(t)), 1/2)
        sigma2 = self.c*torch.pow(1. - torch.exp(self.alpha * self.marginal_log_mean_coeff(t)), 1 / self.alpha)
        return sigma1, sigma2


