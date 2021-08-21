import torch
import torchvision.utils as utils
import io
from PIL import Image
from enum import Enum, auto

from train import loadModel
from generator import DCGenerator, getImage

class GType(Enum):
    CELEBA_30_E = "celeba-30-e"
    CELEBA_20_E = "celeba-20-e"
    CELEBA_10_E = "celeba-10-e"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class GeneratorManager():
    def __init__(self):
        self.generators = {
            GType.CELEBA_30_E.value: loadModel("models/dcgan_celeba/dcgan_celeba_30_g", DCGenerator),
            GType.CELEBA_20_E.value: loadModel("models/dcgan_celeba/dcgan_celeba_20_g", DCGenerator),
            GType.CELEBA_10_E.value: loadModel("models/dcgan_celeba/dcgan_celeba_10_g", DCGenerator)
        }

    def generateImage(self, g_type: GType, image_number, label = None):
        rand_tensor = rand_tensor = torch.randn(64, 100, 1, 1)

        # if label is not None:
        #     out_tensor = self.generators[g_type](rand_tensor, label).squeeze()
        # else:
        out_tensor = self.generators[g_type](rand_tensor).squeeze()

        return self.__tensorToPNG(image_number, out_tensor)

    def __tensorToPNG(self, image_number, out_tensor):
        grid = utils.make_grid(out_tensor[:image_number], padding=2, normalize=True)

        image = Image.fromarray(getImage(grid))
        buffer = io.BytesIO()
        image.save(buffer, 'PNG')
        buffer.seek(0)

        return buffer