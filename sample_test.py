import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

from sampling import *
from training import *
import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity



if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

"""
for initial_clamp in torch.arange(1, 3, 0.2):
    for clamp in torch.arange(1,3,0.2):
        for clamp_mode in ['constant', 'linear']:
            print(f'initial clamp{initial_clamp}, clamp {clamp}, model {clamp_mode}')
            samples = sample(alpha=1.9, path='/scratch/private/eunbiyoon/Levy_motion-/CIFAR10clamp3.01.9_0.1_7.5.pth',
                             beta_min=0.1, beta_max=7.5, sampler='pc_sampler2', batch_size=64, num_steps=1000,
                             LM_steps=10,
                             Predictor=True, Corrector=False, trajectory=False, clamp=clamp, initial_clamp=initial_clamp,
                             clamp_mode=clamp_mode, datasets="CIFAR10")

"""

for epoch in torch.arange(8,9):
  path =f"/scratch/private/eunbiyoon/Levy_motion-/checkpoint/CelebAclamp3.0epoch3_1.9_0.1_7.5.pth"
  samples = sample(alpha=1.9, path="/scratch/private/eunbiyoon/Levy_motion-/checkpoint/CelebAclamp3.0epoch5_1.9_0.1_7.5.pth",
                   beta_min=0.1, beta_max=7.5, sampler='ode_sampler', batch_size=64, num_steps=100, LM_steps=50,
                   Predictor=True, Corrector=False, trajectory=False, clamp=3, initial_clamp=2.0, clamp_mode="constant",
                   datasets="CelebA", name=epoch.item())

#train(beta_min=0.1, alpha=1.9,beta_max=20,n_epochs=1000)
#train(beta_min=1, alpha=1.9, abeta_max=20,n_epochs=1000)


#samples = sample(alpha=1.9, path ='/scratch/private/eunbiyoon/Levy_motion-/CIFAR10clamp3.0epoch41.9_0.1_7.5.pth',
#            beta_min=0.1, beta_max=7.5, sampler='pc_sampler2', batch_size=64, num_steps=1000,LM_steps=10,
#            Predictor=True, Corrector=False, trajectory =False, clamp =3, initial_clamp =3, clamp_mode = "constant", datasets="CIFAR10")
#sample(alpha=1.9, path ="/home/eunbiyoon/Levy_motion-/MNIST1.9_0.1_20.pth",beta_min=0.1, beta_max=20, sampler='pc_sampler2', batch_size=64,num_steps=1000,LM_steps=20,Predictor=True, Corrector=False)

