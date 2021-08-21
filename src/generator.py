import torch
import torch.nn as nn
from numpy import uint8

nout = 784 # Number of output, 28 * 28
nf = 128 # Number of feature maps
ns = 0.2 # Negative slope for LeakyReLU
nz = 100
nc = 3

def getImage(vectors):
    image = vectors.permute(1, 2, 0).detach().numpy()
    return (image * 255).astype(uint8)

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

class CGenerator(nn.Module): # Class to build generator model
    def __init__(self):
        super(CGenerator, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(138, 256), # Transformation linéaire, y[256] = x[nf] * A^T + b
            nn.LeakyReLU(ns), # Fonction d'activiation, minimise les valeurs nég, maximise les valeurs pos
            nn.Linear(256, 512),
            nn.LeakyReLU(ns),
            nn.Linear(512, 1024),
            nn.LeakyReLU(ns),
            nn.Linear(1024, nout),
            nn.Tanh() # Fonction d'activation, met les valeurs entre [-1;1]
        )
        self.label_emb = nn.Embedding(10, 10)

    def forward(self, noise, labels):
        noise = noise.view(noise.shape[0], -1)
        c = self.label_emb(labels)
        input = torch.cat([noise, c], 1)
        out = self.main(input)
        return (out)

class DCGenerator(nn.Module):
    def __init__(self, ngpu):
        super(DCGenerator, self).__init__()
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

# Conditionnal Deep Convolutionnal GAN
class cDCGenerator(nn.Module):
    def __init__(self, ngpu):
        super(cDCGenerator, self).__init__()
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
        self.label_emb = nn.Embedding(10,10)

    def init_weight(self):
        for it in self._modules:
            if isinstance(self._modules[it], nn.ConvTranspose2d):
                self._modules[it].weight.data.normal_(0.0, 0.02)
                self._modules[it].bias.data.zero_()

    def forward(self, input, labels):
        Z = self.label_emb(labels)
        X = torch.cat([input, Z[:, :, None, None]], 1)
        return self.main(X)

# Wassertein Conditionnal Deep Convolutionnal Generator (CelebA)
class WCDC_Generator(nn.Module):
    def __init__(self, ngpu):
        super(WCDC_Generator, self).__init__()
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
        self.label_emb = nn.Embedding(3,10)

    def init_weight(self):
        for it in self._modules:
            if isinstance(self._modules[it], nn.ConvTranspose2d):
                self._modules[it].weight.data.normal_(0.0, 0.02)
                self._modules[it].bias.data.zero_()

    def forward(self, input, labels):
        Z = self.label_emb(labels)
        X = torch.cat([input, Z[:, :, None, None]], 1)
        return self.main(X)
