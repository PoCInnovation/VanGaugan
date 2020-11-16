# VanGaugan

A Deep Convolutional Generative Adversarial Networks implementation using PyTorch.

The aim of the project is to create an Artificial Intelligence
able to generate fake images, such as faces or landscapes,
which appear real from a human point of view.

## Installation

```
$ git clone git@github.com:PoCInnovation/VanGaugan.git
$ pip3 install -r requirements.txt
```

## Train a model

```
$ ./VanGaugan train -e [EPOCH_NUMBER] -g [GENERATOR_MODEL_SAVE_PATH] -d [DISCRIMINATOR_MODEL_SAVE_PATH]
```
GPU instance is required to train.


## Load a model and display a grid of generated images

```
$ ./VanGaugan load [GENERATOR_MODEL_PATH]
```

## Create a GIF to vizualise training progress

```
$ ./VanGaugan gif -o [GIF_OUTPUT_PATH] -md [MODELS_DIRECTORY_PATH]
```

## Serve models through an HTTP API

```
$ ./VanGaugan serve
```

## API endpoints

  - `GET /api/list-models` : list availables models
  - `GET /api/<model_name>?image_number=<image_number>&label=<label>` : generate images
 
 
 ## Run front-end
 
 ```
 $ cd front
 $ npm install
 $ npm run start
 ```
 
 or create production build using `npm run build` and serve it whith what you want.
 
 
 ## Authors
 
 - [Clément Doucy](https://github.com/ClementDoucy/)
 - [Mattis Cusin](https://github.com/Basilarc)
