import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from torch.autograd import Variable
from generator import CGenerator, getImage
from discriminator import CDiscriminator
from sys import argv, exit

BS = 128 # Batch size
LR = 0.0002 # Learning Rate

# Load MNIST DATASET
mnistDataset = dset.MNIST(
        './dataset',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            ])
        )
mnistLoader = torch.utils.data.DataLoader( 
    mnistDataset,
    batch_size=BS, shuffle=True
)

class CTrainer():
    def __init__(self):
        self.GNet = CGenerator()
        self.DNet = CDiscriminator()

        self.GOpti = optim.Adam(self.GNet.parameters(), lr=LR)
        self.DOpti = optim.Adam(self.DNet.parameters(), lr=LR)
        # Adam optimizer -> Stochastic Optimization

        self.loss_fun = torch.nn.BCELoss() # Calcul d'erreur concernant le GAN
        self.writter = SummaryWriter(log_dir='log/loss', comment='Training loss') # logger pour tensorboard

    def __del__(self):
        self.writter.close()

    # Entraine le modèle du generator
    def trainGNet(self, fake_data, fake_labels): #essayons de retirer fake_data
        self.GOpti.zero_grad()
        fake_imgs = self.GNet(torch.randn(128, BS), fake_labels)
        # équivaut à un appel de forward
        validity = self.DNet(fake_imgs, fake_labels)
        g_loss = self.loss_fun(validity, torch.ones(BS, 1))
        g_loss.backward()
        self.GOpti.step()
        return g_loss


    # Entraine le modèle du discriminant
    def trainDNet(self, fake_data, fake_labels, real_data , labels):
        self.DOpti.zero_grad()

        # Entraînement avec des images réelles
        real_validity = self.DNet(real_data, labels)
        real_loss = self.loss_fun(real_validity, torch.ones(real_data.shape[0]))


        # Entraînement avec des images générées
        fake_validity = self.DNet(fake_data, fake_labels)
        fake_loss = self.loss_fun(fake_validity, torch.zeros(BS))

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
                s = batch.size(0)

                # Préparation du set d'images réelles
                real_imgs = self.preprocess(batch, 784)

                # préparation du set d'images générés et de ses labels
                fake_labels = torch.randint(0, 10,(BS,))
                fake_imgs = self.GNet(torch.randn(128, BS), fake_labels).detach()

                # Entraîenement des 2 réseaux sur a bases des images générées
                DResult = self.trainDNet(fake_imgs, fake_labels,real_imgs, labels)
                GError = self.trainGNet(fake_imgs, fake_labels)

                i += 1


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

def load_and_show(path, label):
    GNet = loadModel(path, Generator)
    label_tensor = torch.LongTensor([label])
    rand_tensor = torch.randn(1, 128)
    res = GNet(rand_tensor, label_tensor)
    plt.imshow(getImage(res), cmap='gray')
    plt.show()

if __name__ == "__main__":
    load_and_show("./models/genator_50_e")
    exit(0)
    t = Trainer()
    t(int(argv[1]), mnistLoader)

    t.save(argv[2], argv[3])

    rand_tensor = torch.randn(1, BS) # Create random input

    output = t.GNet(1) # Call generator with random input

    plt.imshow(getImage(output), cmap='gray') # Plot output
    plt.show()