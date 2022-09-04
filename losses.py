import torch
import copy
import time

from scipy.special import gamma
from torchlevy import levy
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

def gamma_fn(x):
    return torch.tensor(gamma(x))



def loss_fn(model, sde,
            x0: torch.Tensor,
            t: torch.LongTensor,
            e_B,
            e_L: torch,
            num_steps=1000, type="cft", mode='only'):

    x_coeff = sde.diffusion_coeff(t)
    sigma1, sigma2 = sde.marginal_std(t)

    score = levy.score(sigma1[:,None,None,None]*e_B+sigma2[:,None,None,None]*e_L, sde.alpha, sigma1 = sigma1, sigma2 = sigma2,type=type).to(device)* sde.beta(t)[:,None,None,None]

    x_t = x_coeff[:, None, None, None] * x0 + e_B*sigma1[:,None,None,None]+ e_L * sigma2[:, None, None, None]
    output = model(x_t, t) * sde.beta(t)[:, None, None, None]
    weight = (output - score)
    loss = (weight).square().sum(dim=(1, 2, 3)).mean(dim=0)

    #print('x_t', torch.min(x_t), torch.max(x_t))
    #print('e_L', torch.min(e_L),torch.max(e_L))
    #print('score', torch.min(score), torch.max(score))
    #print('output', torch.min(model(x_t, t)), torch.max(model(x_t, t)))
    #print('output*beta',torch.min(output), torch.max(output))
    #print('weight', torch.min(weight), torch.max(weight))

    return  loss



