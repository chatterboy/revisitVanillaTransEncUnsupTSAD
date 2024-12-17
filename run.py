import os
import random
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')  # for computing precision and f1 score given 0.0.

import numpy as np

import torch

import wandb

from trainer import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, default="train")

    # Dataset
    parser.add_argument("--data_path", type=str, default="./data/PSM")
    parser.add_argument("--data_name", type=str, default="PSM",
                        help="data name: [ \
                            ABP, Acceleration, AirTemperature, ECG, EPG, Gait, NASA, PowerDemand, RESP, \
                            MSL, SMAP, SMD, PSM, SWAN_SF, GECCO]")
    parser.add_argument('--n_vars', type=int, default=25)
    parser.add_argument('--split_ratio', type=float, default=None)
    parser.add_argument('--norm', type=str, default=None,
                        help='norm name: [None, minmax, standard]')
    # parser.add_argument("--norm_level", type=str, default=None,
    #                     help="norm_level option: [None, entire, separate]")

    # Checkpoint
    parser.add_argument("--ckpt", type=str, default="ckpt")

    # Data embedding
    parser.add_argument("--data_embed", type=str, default="local")
    parser.add_argument("--k_size", type=int, default=3)  # local embedding

    #
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_epochs", type=int, default=100)

    parser.add_argument("--win_size", type=int, default=100)
    parser.add_argument("--step_size", type=int, default=1)
    parser.add_argument("--test_step_size", type=int, default=1)

    # Model
    parser.add_argument("--model_name", type=str, default="Autoencoder",
                        help='model name: [Autoencoder]')
    parser.add_argument("--percentile", type=float, default=None)

    # Model's hyperparameters (Transformer Encoder)
    parser.add_argument("--e_layers", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Ablation study
    parser.add_argument("--pe_mode", type=str, default=None)  # fixed / learnable

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--use_multi_gpu', type=bool, default=False)
    parser.add_argument('--devices', type=str, default='0')  # '0,1' or '0,1,2,3'

    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.seed is None:
        args.seed = int(datetime.now().timestamp())

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.isdir(args.ckpt):
        os.mkdir(args.ckpt)

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = [int(_) for _ in args.devices.split(',')]

    print(args)

    # wandb.init()
    # wandb.config.update(args)

    exp = Trainer(args)

    if args.mode == "train":
        exp.train()
    elif args.mode == "test":
        exp.test()
    else:
        raise ValueError("Expected 'train' or 'test', but got '{}'".format(args.mode))