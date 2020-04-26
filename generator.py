import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd.variable import Variable

nout = 784 # Number of output, 28 * 28
nf = 128 # Number of feature maps
ns = 0.2 # Negative slope for LeakyReLU

def getImage(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28).data[0][0] # Convert vector generator output to 28 * 28 image

class Generator(nn.Module): # Class to build generator model
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nf, 256), # Transformation linéaire, y[256] = x[nf] * A^T + b
            nn.LeakyReLU(ns), # Fonction d'activiation, minimise les valeurs nég, maximise les valeurs pos
            nn.Linear(256, 512),
            nn.LeakyReLU(ns),
            nn.Linear(512, 1024),
            nn.LeakyReLU(ns),
            nn.Linear(1024, nout),
            nn.Tanh() # Fonction d'activation, met les valeurs entre [-1;1]
        )

    def forward(self, input):
        return self.main(input)

if __name__ == "__main__":
    netGenerator = Generator() # Build generator

    rand_tensor = Variable(torch.randn(1, nf)) # Create random input

    output = netGenerator(rand_tensor) # Call generator with random input

    plt.imshow(getImage(output), cmap='gray') # Plot output
    plt.show()