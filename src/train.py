import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
#from torch.utils.tensorboard import SummaryWriter

from generator import Generator, getImage, CGenerator
from discriminator import Discriminator, CDiscriminator
from sys import argv, exit

BS = 128 # Batch size
LR = 0.0002 # Learning Rate
IMG_SIZE = 64

mnistLoader = torch.utils.data.DataLoader( # Load MNIST DATASET
    dset.MNIST(
        './dataset',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])
        ),
    batch_size=BS, shuffle=True
)

class Trainer():
    def __init__(self, ngpu):

        print(torch.cuda.is_available())
        device_type = "cuda:0" if torch.cuda.is_available() and ngpu > 0 else "cpu"
        self.device = torch.device(device_type)

        self.GNet = CGenerator(ngpu).to(self.device)
        self.DNet = CDiscriminator(ngpu).to(self.device)

        print(device_type)
        print(self.device.type)
        if self.device.type == "cuda" and ngpu > 1:
            device_ids = list(range(ngpu))
            self.GNet = nn.DataParallel(self.GNet, device_ids=device_ids)
            self.DNet = nn.DataParallel(self.DNet, device_ids=device_ids)
            print("GPU OK")

        self.GNet.init_weight()
        self.DNet.init_weight()

        self.GOpti = optim.Adam(self.GNet.parameters(), lr=LR)
        self.DOpti = optim.Adam(self.DNet.parameters(), lr=LR)
        # Adam optimizer -> Stochastic Optimization

        self.lossFun = nn.BCELoss() # Binary cross entropy, Prend 2 paramètres
 #       self.writter = SummaryWriter(log_dir='log/loss', comment='Training loss') # logger pour tensorboard


    def __del__(self):
        None
        #self.writter.close()

    # Entraine le modèle du generator
    def trainGNet(self, fakeData, size):
        # Call discriminator with fake data to compute generator loss
        self.GOpti.zero_grad()
        result = self.DNet(fakeData).squeeze()

        expected = torch.ones(size, device=self.device)
        err = self.lossFun(result, expected)
        err.backward()

        self.GOpti.step()
        return err


    # Entraine le modèle du discriminant
    def trainDNet(self, realData, fakeData, size):
        self.DOpti.zero_grad()

        # Train with real data
        realRes = self.DNet(realData).squeeze()
        expected = torch.ones(size, device=self.device)
        realErr = self.lossFun(realRes, expected)
        realErr.backward()

        # Train with fake data
        fakeRes = self.DNet(fakeData).squeeze()
        expected = torch.zeros(size, device=self.device)
        fakeErr = self.lossFun(fakeRes, expected)
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
                print(f"iteration = {i}")
                real = batch.to(self.device)
                size = real.size(0)
                fake = self.GNet(self.createNoise(size))
                DResult = self.trainDNet(real, fake.detach(), size)

                fake = self.GNet(self.createNoise(size))
                GError = self.trainGNet(fake, size)

            self.log(e, DResult['error'], GError)


    def log(self, epoch, DLoss, GLoss):
        print(f"epoch: {epoch}")
        print(f"Discriminator Loss : {DLoss}")
        print(f"Generator Loss : {GLoss}")
        print("==========================================")
  #      self.writter.add_scalar('Loss/Generator', GLoss, epoch)
   #     self.writter.add_scalar('Loss/Discriminator', DLoss, epoch)
    #    self.writter.add_scalars('Loss/Generator+Discriminator', {
     #       'Generator': GLoss,
      #      'Discriminator': DLoss
       # }, epoch)

    def save(self, Gpath, Dpath):
        torch.save(self.GNet.state_dict(), Gpath)
        torch.save(self.DNet.state_dict(), Dpath)

    # renvoie un vecteur normalisé de shape (1, BS)input pour le generator
    def createNoise(self, n):
        return torch.randn(n, 100, 1, 1, device=self.device)

    def preprocess(self, rawData, nout):
        return rawData.view(rawData.size(0), nout)

    def reveal(self, data, i, j):
        return data.view(data.size(0), 1, i, j)


def loadModel(path, Model):
    model = Model()
    try:
        model.load_state_dict(torch.load(path))
    except Exception as error:
        exit(f"Error : {path} : {error}")
    return model

def load_and_show(path):
    GNet = loadModel(path, CGenerator)
    rand_tensor = torch.randn(1, 100).view(-1, 100, 1, 1)
    res = GNet(rand_tensor).squeeze()
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
