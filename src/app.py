from flask import Flask, Response, send_file, request, abort
import torch
import torchvision.utils as utils
import io
from PIL import Image

from train import loadModel
from generator import CGenerator, getImage

HOST = "0.0.0.0"
PORT = 8080

app = Flask(__name__)
generator = None

def run_model_api(modelPath):
    global generator

    generator = loadModel(modelPath, CGenerator)

    app.run(host=HOST, port=PORT)

@app.route("/", methods=["GET"])
def predict():
    image_number = 1

    if request.args.get("image_number") is not None:
        try:
            image_number = int(request.args["image_number"])
            assert image_number >= 1 and image_number <= 64
        except:
            return abort(400, "Invalid image number")

    rand_tensor = torch.randn(64, 100, 1, 1)
    output = generator(rand_tensor).squeeze()
    grid = utils.make_grid(output[:image_number], padding=2, normalize=True)

    image = Image.fromarray(getImage(grid))
    buffer = io.BytesIO()
    image.save(buffer, 'PNG')
    buffer.seek(0)

    return send_file(buffer, mimetype="image/PNG")