from flask import Flask, Response, send_file, request, abort, jsonify
import torch
import torchvision.utils as utils
import io
from PIL import Image
from enum import Enum, auto

from train import loadModel
from generator import CGenerator, getImage
from GeneratorManager import GeneratorManager, GType

HOST = "0.0.0.0"
PORT = 8080

app = Flask(__name__, static_folder='/html', static_url_path='/')
GManager = GeneratorManager()

def run_model_api():
    app.run(host=HOST, port=PORT)

@app.route("/")
def index():
    return app.send_static_file('index.html')

@app.route("/api/<model_type>", methods=["GET"])
def predict(model_type):
    if not GType.has_value(model_type):
        return abort(404, f"{model_type} model doesn't exist")

    image_number = 1
    label = request.args.get("label")

    if request.args.get("image_number") is not None:
        try:
            image_number = int(request.args["image_number"])
            assert image_number >= 1 and image_number <= 64
        except:
            return abort(400, "Invalid image number")

    image = GManager.generateImage(model_type, image_number, label)

    return send_file(image, mimetype="image/PNG")

@app.route("/api/list-models", methods=["GET"])
def list_models():
    return jsonify(list(map(lambda it: it.value, list(GType))))