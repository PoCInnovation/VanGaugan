import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

nout = 784 # Number of output, 28 * 28
nf = 128 # Number of feature maps
ns = 0.2 # Negative slope for LeakyReLU
nz = 100
nc = 1

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


class CGenerator(nn.Module):
    def __init__(self, ngpu):
        super(CGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, nf * 8, 4, 1, 0),
            nn.BatchNorm2d(nf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 8, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1),
            nn.BatchNorm2d(nf),
            nn.ReLU(True),
            nn.ConvTranspose2d(nf, nc, 4, 2, 1),
            nn.Tanh()
        )

    def init_weight(self):
        for it in self._modules:
            if isinstance(self._modules[it], nn.ConvTranspose2d):
                self._modules[it].weight.data.normal_(0.0, 0.02)
                self._modules[it].bias.data.zero_()

    def forward(self, input):
        return self.main(input)

# with SummaryWriter(log_dir='log/generator', comment='Generator network') as sw:
#     sw.add_graph(CGenerator(), torch.randn(1, nf))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    netGenerator = Generator() # Build generator

    rand_tensor = torch.randn(1, nf) # Create random input

    output = netGenerator(rand_tensor) # Call generator with random input

    plt.imshow(getImage(output), cmap='gray') # Plot output
    plt.show()