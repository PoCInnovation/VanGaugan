import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image
import imageio
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
#from torch.utils.tensorboard import SummaryWriter

from generator import Generator, getImage, CGenerator
from discriminator import Discriminator, CDiscriminator
from sys import argv, exit, stderr
from datetime import date, datetime
from os import listdir

BS = 128 # Batch size
LR = 0.0002 # Learning Rate
IMG_SIZE = 64

def loadMnistDataset():
    return torch.utils.data.DataLoader( # Load MNIST DATASET
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

def loadDataset(datasetPath):
    return torch.utils.data.DataLoader(
        dset.ImageFolder(
            root=datasetPath,
            transform=transforms.Compose([
                transforms.Resize(IMG_SIZE),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        ),
        batch_size=BS,
        shuffle=True,
        num_workers=2
    )

class Trainer():
    def __init__(self, ngpu):
        print(torch.cuda.is_available(), file=stderr)
        device_type = "cuda:0" if torch.cuda.is_available() and ngpu > 0 else "cpu"
        self.device = torch.device(device_type)

        self.GNet = CGenerator(ngpu).to(self.device)
        self.DNet = CDiscriminator(ngpu).to(self.device)

        print(device_type, file=stderr)
        print(self.device.type, file=stderr)
        if self.device.type == "cuda" and ngpu > 1:
            device_ids = list(range(ngpu))
            self.GNet = nn.DataParallel(self.GNet, device_ids=device_ids)
            self.DNet = nn.DataParallel(self.DNet, device_ids=device_ids)
            print("GPU OK", file=stderr)

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
                real = batch.to(self.device)
                size = real.size(0)
                fake = self.GNet(self.createNoise(size))
                DResult = self.trainDNet(real, fake.detach(), size)

                fake = self.GNet(self.createNoise(size))
                GError = self.trainGNet(fake, size)
            self.log(e, DResult['error'], GError)
            print(f"Epoch {e + 1} done", file=stderr)
            self.save("./models/default/" + str(date.today()) + "_g_" + str(e + 1),
                "./models/default/" + str(date.today()) + "_d_" + str(e + 1))


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
    model = Model(0)
    try:
        model.load_state_dict(torch.load(path, map_location='cpu'))
    except Exception as error:
        exit(f"Error : {path} : {error}")
    return model

def load_and_show(path):
    GNet = loadModel(path, CGenerator)
    rand_tensor = torch.randn(1, 100, 1, 1)
    res = GNet(rand_tensor).squeeze()
    img = getImage(res)
    plt.imshow(img)
    plt.show()

def make_grid(modelPath):
    Gnet = loadModel(modelPath, CGenerator)
    rand_tensor = torch.randn(64, 100, 1, 1)
    output = Gnet(rand_tensor).squeeze()
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated images")
    grid = utils.make_grid(output, padding=2, normalize=True)
    image = ((grid.permute(1, 2, 0).detach().numpy()))
    plt.imshow(image)
    plt.show()
    save_image(grid, f"{modelPath.split('/').pop()}{datetime.now()}.png")

def createTrainGif(dirPath, gifPath):
    models = listdir(dirPath)
    models.sort(key=lambda it: int(it.split("_").pop()))
    images = []
    rand_tensor = torch.randn(64, 100, 1, 1)

    for fp in models:
        print(f"Predicting with model {fp}...")
        GNet = loadModel(f"{dirPath}/{fp}", CGenerator)
        output = GNet(rand_tensor).squeeze()
        grid = utils.make_grid(output, padding=2, normalize=True)
        images.append(getImage(grid))
    imageio.mimsave(gifPath, images, format="GIF", fps=3)
    print(f"gife saved as {gifPath}")

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
