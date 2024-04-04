import argparse
import os
import random

import numpy as np
import torch

from generate import generate
from train import train
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', type=str, default='generate',
                        help='name of this task: train/generate', required=False)
    parser.add_argument('--run_name', type=str,
                        help="name for an experiment run", required=False)
    parser.add_argument('--data_split', type=str, default='simple',
                        help="data split of SCAN dataset", required=False)
    parser.add_argument('--n_layer', type=int, default=2,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=2,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=16,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=60,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch size", required=False)
    parser.add_argument('--num_workers', type=int, default=1,
                        help="number of workers for data loaders", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=4e-4, help="learning rate", required=False)
    parser.add_argument('--max_len', type=int, default=48,#  to accommodate the longest action sequence observed in the dataset we change it from 128 to 48
                        help="max_len", required=False) # a better for a larger dataset practice is to calculate the median length of the sequence disurbution in the dataset #
    # Add block_size argument to specify the maximum sequence length the model can handle.
    # This is important because transformer models process input data in blocks of a fixed size.
    # If an input sequence exceeds this block size, the model won't be able to process it, leading to errors.
    # By specifying the block_size, we can ensure that all input sequences are either truncated or split to fit within this block size before they're passed to the model.
    parser.add_argument('--block_size', type=int, default=48,
                    help="Maximum sequence length the model can handle", required=False) #
    parser.add_argument('--seed', type=int, default=44,
                        help="seed", required=False)
    parser.add_argument('--grad_norm_clip', type=float, default=1.0,
                        help="gradient norm clipping. smaller values mean stronger normalization.", required=False)
    parser.add_argument('--output_tokenizer_dir',
                        default='./tokenizer',
                        help="Path to the saved tokenizer directory", required=False)

    ### YOUR CODE HERE ###
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if torch.cuda.is_available():
        print("GPU is enabled.")
        
    else:
        print("GPU is not available.")
    ### YOUR CODE HERE ###

    args = parser.parse_args()
    args.ckpt_path = f'./cond_gpt/weights/{args.run_name}_{args.data_split}split_{args.n_layer}layer_{args.n_head}head_{args.n_embd}embd_{args.batch_size}bs.pt'

    set_seed(args.seed)

    if args.task == 'train':
        train(args)
    elif args.task == 'generate':
        generate(args)
    else:
        raise ValueError("Invalid task")
