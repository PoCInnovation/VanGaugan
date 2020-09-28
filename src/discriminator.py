import torch
import torch.nn as nn

nf = 128 # features number
nout = 1 # 1 output : binaiy output
ns = 0.2 # Negative slope for LeakyRelu
p = 0.3 # Dropout layer porbability
nc = 3 # number of c
n_classes = 10 # class number

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nf, 1024),
            nn.LeakyReLU(ns),
            nn.Dropout(p), # Met à 0 des élements aléatoirement, Distribution de Bernoulli
            nn.Linear(1024, 512),
            nn.LeakyReLU(ns),
            nn.Dropout(p),
            nn.Linear(512, 256),
            nn.LeakyReLU(ns),
            nn.Dropout(p),
            nn.Linear(256, nout),
            nn.Sigmoid() # Activation Sigmoid : met les valeur entre 0 et 1
        )

    def forward(self, input):
        return self.main(input)

class CDiscriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nf + 10, 1024),
            nn.LeakyReLU(ns),
            nn.Dropout(p), # Met à 0 des élements aléatoirement, Distribution de Bernoulli
            nn.Linear(1024, 512),
            nn.LeakyReLU(ns),
            nn.Dropout(p),
            nn.Linear(512, 256),
            nn.LeakyReLU(ns),
            nn.Dropout(p),
            nn.Linear(256, 128),
            nn.Dropout(p),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
        # Changer le format des labels
        self.label_emb = nn.Embedding(10, 10)

    #Retourne la validité (ou non) de l'image pour le label spécifié
    def forward(self, input, labels):
        input = input.view(input.shape[0], 784)
        c = self.label_emb(labels)
        input = torch.cat([input, c], 1)
        out = self.main(input)
        return (out.squeeze())

class DCDiscriminator(nn.Module):
    def __init__(self, ngpu):
        super(CDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, nf, 4, 2, 1),
            nn.LeakyReLU(ns),
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(ns),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(ns),
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(ns),
            nn.Conv2d(nf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

    def init_weight(self):
        for it in self._modules:
            if isinstance(self._modules[it], nn.Conv2d):
                self._modules[it].weight.data.normal_(0.0, 0.02)
                self._modules[it].bias.data.zero_()

# Conditionnal Deep Convolutionnal GAN
class CDCDiscriminator(nn.Module):
    def __init__(self, ngpu):
        super(CDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, nf, 4, 2, 1),
            nn.LeakyReLU(ns),
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(ns),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(ns),
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(ns),
            nn.Conv2d(nf * 8, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, input, labels):
        X = torch.cat([input, labels], 1)
        return self.main(X)

    def init_weight(self):
        for it in self._modules:
            if isinstance(self._modules[it], nn.Conv2d):
                self._modules[it].weight.data.normal_(0.0, 0.02)
                self._modules[it].bias.data.zero_()

# Wassertein Conditionnal Deap Convolutionnal Discriminator (CelebA)
class WCDC_Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(WCDC_Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(nc, nf, 4, 2, 1),
            nn.LeakyReLU(ns),
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.LeakyReLU(ns),
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.LeakyReLU(ns),
            nn.Conv2d(nf * 4, nf * 8, 4, 2, 1),
            nn.BatchNorm2d(nf * 8),
            nn.LeakyReLU(ns),
            nn.Conv2d(nf * 8, 1, 4, 1, 0)
        )
    def forward(self, input, labels):
        X = torch.cat([input, labels], 1)
        output = self.main(X)
        output = output.mean(0)
        return output.view(1)

    def init_weight(self):
        for it in self._modules:
            if isinstance(self._modules[it], nn.Conv2d):
                self._modules[it].weight.data.normal_(0.0, 0.02)
                self._modules[it].bias.data.zero_()
