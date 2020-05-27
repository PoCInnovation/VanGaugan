import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from generator import Generator, getImage
from discriminator import Discriminator
from sys import argv, exit

BS = 128 # Batch size
LR = 0.0002 # Learning Rate

mnistLoader = torch.utils.data.DataLoader( # Load MNIST DATASET
    dset.MNIST(
        './dataset',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])
        ),
    batch_size=BS, shuffle=True
)

class Trainer():
    def __init__(self):
        self.GNet = Generator()
        self.DNet = Discriminator()

        self.GOpti = optim.Adam(self.GNet.parameters(), lr=LR)
        self.DOpti = optim.Adam(self.DNet.parameters(), lr=LR)
        # Adam optimizer -> Stochastic Optimization

        self.lossFun = nn.BCELoss() # Binary cross entropy, Prend 2 paramètres
        self.writter = SummaryWriter(log_dir='log/loss', comment='Training loss') # logger pour tensorboard


    def __del__(self):
        self.writter.close()

    # Entraine le modèle du generator
    def trainGNet(self, fakeData):
        self.GOpti.zero_grad()
        result = self.DNet(fakeData)

        sizeAvrg = torch.ones(fakeData.size(0), 1)
        err = self.lossFun(result, sizeAvrg)
        err.backward()

        self.GOpti.step()
        return err


    # Entraine le modèle du discriminant
    def trainDNet(self, realData, fakeData):
        self.DOpti.zero_grad()
        realRes = self.DNet(realData)
        sizeAvrg = torch.ones(fakeData.size(0), 1)
        realErr = self.lossFun(realRes, sizeAvrg)
        realErr.backward()

        fakeRes = self.DNet(fakeData)
        sizeAvrg = torch.zeros(fakeData.size(0), 1)
        fakeErr = self.lossFun(fakeRes, sizeAvrg)
        fakeErr.backward()

        self.DOpti.step()
        return {
            "error": realErr + fakeErr,
            "realRes": realRes,
            "fakeRes": fakeRes
        }

    def __call__(self, epoch, loader):
        for e in range(epoch):
            for i, (batch, _) in enumerate(loader):
                print("iteration = ", i)
                s = batch.size(0)

                real = self.preprocess(batch, 784)
                fake = self.GNet(self.createNoise(s))
                DResult = self.trainDNet(real, fake.detach())

                # fake = self.GNet(self.createNoise(s))
                GError = self.trainGNet(fake)


            self.log(e, DResult['error'], GError)


    def log(self, epoch, DLoss, GLoss):
        print(f"epoch: {epoch}")
        print(f"Discriminator Loss : {DLoss}")
        print(f"Generator Loss : {GLoss}")
        print("==========================================")
        self.writter.add_scalar('Loss/Generator', GLoss, epoch)
        self.writter.add_scalar('Loss/Discriminator', DLoss, epoch)
        self.writter.add_scalars('Loss/Generator+Discriminator', {
            'Generator': GLoss,
            'Discriminator': DLoss
        }, epoch)

    def save(self, Gpath, Dpath):
        torch.save(self.GNet.state_dict(), Gpath)
        torch.save(self.DNet.state_dict(), Dpath)

    # renvoie un vecteur normalisé de shape (1, BS)input pour le generator
    def createNoise(self, n):
        return torch.randn(n, BS)

    def preprocess(self, rawData, nout):
        return rawData.view(rawData.size(0), nout)

    def reveal(self, data, i, j):
        return data.view(data.size(0), 1, i, j)


def loadModel(path, Model):
    model = Model()
    try:
        model.load_state_dict(torch.load(path))
    except:
        exit(f"Error : {path} : invalid model path.")
    return model

def load_and_show(path):
    GNet = loadModel(path, Generator)
    rand_tensor = torch.randn(1, BS)
    print(rand_tensor.mean())
    res = GNet(rand_tensor)
    plt.imshow(getImage(res), cmap='gray')
    plt.show()

if __name__ == "__main__":
    load_and_show("./models/genator_50_e")
    exit(0)
    t = Trainer()
    t(int(argv[1]), mnistLoader)

    t.save(argv[2], argv[3])

    rand_tensor = torch.randn(1, BS) # Create random input

    output = t.GNet(rand_tensor) # Call generator with random input

    plt.imshow(getImage(output), cmap='gray') # Plot output
    plt.show()
