import torch
import torchvision.utils as utils
import io
from PIL import Image
from enum import Enum, auto

from train import loadModel
from generator import CGenerator, getImage

class GType(Enum):
    CELEBA = "celeba"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class GeneratorManager():
    def __init__(self):
        self.generators = {
            GType.CELEBA.value: loadModel("models/celeba_30_g", CGenerator)
        }

    def generateImage(self, g_type: GType, image_number, label = None):
        rand_tensor = rand_tensor = torch.randn(64, 100, 1, 1)

        if label is not None and g_type != "celeba":
            out_tensor = self.generators[g_type](rand_tensor, label).squeeze()
        else:
            out_tensor = self.generators[g_type](rand_tensor).squeeze()

        return self.__tensorToPNG(image_number, out_tensor)

    def __tensorToPNG(self, image_number, out_tensor):
        grid = utils.make_grid(out_tensor[:image_number], padding=2, normalize=True)

        image = Image.fromarray(getImage(grid))
        buffer = io.BytesIO()
        image.save(buffer, 'PNG')
        buffer.seek(0)

        return buffer