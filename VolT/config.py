import os
import json
import argparse


def gen_config():
    parser = argparse.ArgumentParser()

    # parameters on training and validating


    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--generator_iters', type=int, default=80000)
    parser.add_argument('--LAMBDA', type=int, default=10)
    parser.add_argument('--SAVE_PER_TIMES', type=int, default=100)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--model_name', type=str, default='VolT_WGAN')
    parser.add_argument('--checkpoint_save_path', type=str, default='checkpoints/VolT/')

    args = parser.parse_args()

    return args


def save_config(args, save_path):
    with open(save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_config(load_path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(load_path, 'r') as f:
        args.__dict__ = json.load(f)
    return args


if __name__ == '__main__':
    parser = gen_config()
    print(parser.epochs)
    # save_config(parser, 'config.json')
    # config = load_config('config.json')
