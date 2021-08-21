from argparse import ArgumentParser
from sys import exit, argv, stderr, path
path.append("./src")
from train import Trainer, load_and_show, make_grid, createTrainGif, loadDataset
from wtrain import gen_fake_labels
from app import run_model_api
import random

CELEBA_DIR="dataset/CelebA/"
ARTWORKS_DIR="dataset/artworks/"
WIKIART="dataset/wikiart"

def parseArgs():
    parser = ArgumentParser()
    sub = parser.add_subparsers()

    parser_1 = sub.add_parser("load", help="load model and show a fake generated image")
    parser_1.add_argument("filepath", type=str, help="path of the model to load")

    parser_2 = sub.add_parser("train" , help="train generator and discriminator")
    parser_2.add_argument("-e", "--epoch", type=int, help="epoch number")
    parser_2.add_argument("-g", "--generator", help="generator model output file")
    parser_2.add_argument("-d", "--discriminator", help="discriminator model output file")
    parser_2.add_argument("-n", "--ngpu", type=int, help="number of GPU to use", default=0)
    parser_2.add_argument("-l", "--label", help="Image label to use for generation")

    parser_3 = sub.add_parser("gif", help="create a training gif")
    parser_3.add_argument("-md", "--models_dir", type=str, help="directory with models")
    parser_3.add_argument("-o", "--output", type=str, help="gif output")

    parser_4 = sub.add_parser("serve", help="serve model using HTTP api")
    parser_4.add_argument("-gp", "--generator_path", type=str, help="generator file path")

    return parser.parse_args()

def main():
    if len(argv) < 2:
        print("Error : no arguments provided.", file=stderr)
        return 84
    args = parseArgs()
    if "epoch" in args:
        t = Trainer(args.ngpu)
        t(args.epoch, loadDataset(CELEBA_DIR)) # CustomCelebAdataset() to use with wtrainer
        t.save(args.generator, args.discriminator)
        del t
    elif "filepath" in args:
        make_grid(args.filepath)
        # For wgan
        # if args.label != None :
        #     load_and_show(args.filepath, int(args.label) if args.label != None else random.randint(0, 9))
        # else :
        #     # generate random labels
        #     fake_labels = gen_fake_labels(1)
        #     load_and_show(args.filepath, fake_labels)
    elif "models_dir" in args:
        createTrainGif(args.models_dir, args.output)
    elif "generator_path" in args:
        run_model_api()
    return 0

if __name__ == "__main__":
    exit(main())