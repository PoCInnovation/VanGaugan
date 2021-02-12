# VanGaugan

Multiple Convolutionnal Generative Adversarial networks implementations using Pytorch.

The aim of the project is to create an Artificial Intelligence
able to generate fake images, such as faces or landscapes,
which appear real from a human point of view.

## Installation

You can easily test the Deep Convolutional Generative Adversarial Network implementation.

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

You can also use Docker : please take a look to [Docker](#Docker) section.

## Docker

```bash
docker-compose up
```

and wait a bit. You should be able to reach front-end [here](http://localhost). It looks like this :



![](/home/matthis/Documents/Epitech/POC/VanGaugan/Pictures/front.png)

*front-end screenshot*

<u>N.B. :</u> Labels field has been added for a futur iteration

### Docker architecture

![](/home/matthis/Documents/Epitech/POC/VanGaugan/Pictures/DockerArchi.png)

*Schematic representation of VanGaugan docker architecture*

## Authors

- [Clément Doucy](https://github.com/ClementDoucy/)
- [Matthis Cusin](https://github.com/Basilarc)
