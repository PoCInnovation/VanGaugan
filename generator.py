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

test_loader = torch.utils.data.DataLoader(
    dset.MNIST(
        './dataset',
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])
        ),
    batch_size=64, shuffle=True
)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

nout = 784 # Number of output, 28 * 28
nf = 100 # Number of feature maps


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28).data[0][0] # Convert vector generator output to 28 * 28 image

class Generator(nn.Module): # Class to build generator model
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nf, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, nout),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

netGenerator = Generator() # Build generator

rand_tensor = Variable(torch.randn(1, 100)) # Create random input

output = netGenerator(rand_tensor) # Call generator with random input

plt.imshow(vectors_to_images(output), cmap='gray') # Plot output
plt.show()