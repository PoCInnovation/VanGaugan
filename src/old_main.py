import random
from argparse import ArgumentParser
from sys import exit, argv, stderr, path
path.append("./src")
from train import Trainer, load_and_show, mnistLoader

def parseArgs():
    parser = ArgumentParser()
    sub = parser.add_subparsers()

    parser_1 = sub.add_parser("load", help="load model and show a fake generated image")
    parser_1.add_argument("filepath", type=str, help="path of the model to load")
    parser_1.add_argument("-l", "--label", help="Image label to use for generation")

    parser_2 = sub.add_parser("train" , help="train generator and discriminator")
    parser_2.add_argument("-e", "--epoch", type=int, help="epoch number")
    parser_2.add_argument("-g", "--generator", help="generator model output file")
    parser_2.add_argument("-d", "--discriminator", help="discriminator model output file")

    return parser.parse_args()

def main():
    if len(argv) < 2:
        print("Error : no arguments provided.", file=stderr)
        return 84
    args = parseArgs()
    if "epoch" in args:
        t = Trainer()
        t(args.epoch, mnistLoader)
        t.save(args.generator, args.discriminator)
        del t
    elif "filepath" in args:
        if "label" in args :
            load_and_show(args.filepath, int(args.label) if args.label != None else random.randint(0, 9))
        else :
            load_and_show(args.filepath, random.randint(0, 9))
    return 0


if __name__ == "__main__":
    exit(main())