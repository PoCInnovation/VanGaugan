import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from celeba_dataset import CelebaDataset
from pathlib import Path

from generator import Generator, getImage, CGenerator, cDCGenerator
from discriminator import Discriminator, CDiscriminator, cDCDiscriminator
from sys import argv, exit, stderr
from datetime import date

BS = 128 # Batch size
LR = 0.0002 # Learning Rate
IMG_SIZE = 64
N_CLASSES = 10

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

class CDCTrainer():
    def __init__(self, ngpu):
        print(torch.cuda.is_available())
        device_type = "cuda:0" if torch.cuda.is_available() and ngpu >= 0 else "cpu"
        self.device = torch.device(device_type)

        self.GNet = cDCGenerator(ngpu).to(self.device)
        self.DNet = cDCDiscriminator(ngpu).to(self.device)

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

        self.loss_fun = torch.nn.BCELoss() # Error calculation concerning GAN
        self.writter = SummaryWriter(log_dir='log/loss', comment='Training loss') # logger pour tensorboard

        self.fill = torch.zeros(10, 10, IMG_SIZE, IMG_SIZE, device=self.device)
        for i in range(0, N_CLASSES):
            self.fill[i, i , :, :] = 1

    def __del__(self):
        self.writter.close()

    # Train generator model
    def trainGNet(self):
        self.GOpti.zero_grad()

        fake_labels = gen_fake_labels(BS, self.device)
        fake_labels_fill = self.fill[fake_labels]
        
        fake_imgs = self.GNet(self.createNoise(fake_labels.size(0)), fake_labels)
        validity = self.DNet(fake_imgs, fake_labels_fill)

        g_loss = self.loss_fun(validity, torch.ones(BS, 1, 1, 1, device=self.device))
        g_loss.backward()

        self.GOpti.step()
        return g_loss


    # Train discriminator model
    def trainDNet(self, fake_data, fake_labels, real_data , labels):
        self.DOpti.zero_grad()

        labels_fill = self.fill[labels]
        fake_labels_fill = self.fill[fake_labels]

        # Train with real pictures
        real_validity = self.DNet(real_data, labels_fill)
        real_loss = self.loss_fun(real_validity, torch.ones(real_data.shape[0], 1, 1, 1, device=self.device))

        # Train with generated pictures
        fake_validity = self.DNet(fake_data, fake_labels_fill)
        fake_loss = self.loss_fun(fake_validity, torch.zeros(BS, 1, 1, 1, device=self.device))

        d_loss = real_loss + fake_loss
        d_loss.backward()

        self.DOpti.step()
        return {
            "error": d_loss,
            "realRes": real_validity,
            "fakeRes": fake_validity
        }

    def __call__(self, epoch, loader):
        for e in range(epoch):
            i = 0
            for i, (batch, labels) in enumerate(loader):
                print("iteration = ", i)

                # Transform batch in order to make it use the right device and get his real size
                real_imgs = batch.to(self.device)
                s = real_imgs.size(0)

                # Prepare generated pictures and lables sets
                fake_labels = gen_fake_labels(BS, self.device)
                fake_imgs = self.GNet(self.createNoise(BS), fake_labels).detach()

                DResult = self.trainDNet(fake_imgs, fake_labels, real_imgs, labels.cuda() if torch.cuda.is_available() else labels)
                for j in range(0, 2) :
                    GError = self.trainGNet()

            self.log(e, DResult['error'], GError)
            print(f"Epoch {e + 1} done", file=stderr)
            self.save("./models/default/" + str(date.today()) + "_g_" + str(e + 1),
                "./models/default/" + str(date.today()) + "_d_" + str(e + 1))

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

    # Return a normalized vector of shape (1, BS)used as input generator
    def createNoise(self, n):
        return torch.randn(n, 100, 1, 1, device=self.device)

    def preprocess(self, rawData, nout):
        return rawData.view(rawData.size(0), nout)

    def reveal(self, data, i, j):
        return data.view(data.size(0), 1, i, j)

def gen_fake_labels(n, device='cpu') :
    fake_labels = torch.randint(0, 10, (n,) , device=device)
    return (fake_labels)

def loadModel(path, Model):
    model = Model(0)
    try:
        model.load_state_dict(torch.load(path, map_location='cpu'))
    except Exception as error:
        exit(f"Error : {path} : {error}")
    return model

def load_and_show(path, label):
    GNet = loadModel(path, CGenerator)
    rand_tensor = torch.randn(1, 100, 1, 1)
    res = GNet(rand_tensor, label).squeeze()
    img = getImage(res)
    plt.imshow(img)
    plt.show()

def make_grid(modelPath):
    Gnet = loadModel(modelPath, CGenerator)
    rand_labels = gen_fake_labels(64)
    rand_tensor = torch.randn(64, 100, 1, 1)
    output = Gnet(rand_tensor, rand_labels).squeeze()
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Generated images")
    grid = utils.make_grid(output, padding=2, normalize=True)
    image = ((grid.permute(1, 2, 0).detach().numpy()))
    plt.imshow(image)
    plt.show()
