import torch
import torch.nn as nn

nf = 784 # nombre de features : 784 pixels (28 * 28)
nout = 1 # 1 output : sortie binaire
ns = 0.2 # Negative slope pour LeakyRelu
p = 0.3 # probabilité pour Dropout layer

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

if __name__ == "__main__":
    from generator import Generator
    G = Generator()
    r = torch.randn(1, 128)
    y = G(r)
    D = Discriminator()
    y_ = D(y)
    print(y_)