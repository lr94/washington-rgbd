#!/usr/bin/env python3
import argparse
import tqdm
import torch
from torch.utils.data.dataloader import DataLoader

from utils import *
from loader import init_washington_datasets


def main():
    args = parse_args()

    full_dataset = init_washington_datasets(args.rgb_root, train_split=args.split, dataset_type='rgb')

    device, workers, _, batch_size, _ = parse_device_model_args(args)

    result = compute_avg_std(full_dataset, batch_size=args.batch_size, workers=workers, device=device)

    print("Mean: ", result[0])
    print("Std:  ", result[1])


@stopwatch
def compute_avg_std(dataset, batch_size=64, workers=1, device=None):
    """
    Compute dataset average and standard deviation using an online algorithm (Welford, 1962)
    Both statistics are computed for each channel
    :param dataset:
    :param batch_size:
    :param device: None, torch.device('cpu') or torch.device('cuda')
    :return: A tuple containing mean and standard deviation
    """

    loader = DataLoader(dataset, batch_size, num_workers=workers)

    mean = torch.zeros(3, device=device)
    m2 = torch.zeros(3, device=device)

    n = 0

    pb = tqdm(total=len(dataset), unit=" samples")

    for batch, _ in loader:
        if device is not None:
            batch = batch.to(device)

        b, c, h, w = batch.shape
        pixels = b * h * w
        n += pixels
        delta = batch - mean[None, :, None, None]
        mean += torch.sum(delta, dim=(0, 2, 3)) / n
        m2 += torch.sum(delta * (batch - mean[None, :, None, None]), dim=(0, 2, 3))

        pb.update(b)

    if n < 2:
        return mean, torch.tensor(float('nan'))  # This should never happen
    else:
        return mean, torch.sqrt(m2 / (n - 1))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Program to compute mean and standard deviation of the dataset",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100)
    )

    dataset_opt_g = parser.add_argument_group(title="Dataset options")
    dataset_opt_g.add_argument('--rgb-root', help="Folder containing the dataset")

    dataset_opt_g.add_argument('--split', help="Dataset split (.txt) to use")

    add_device_options(parser)

    args = parser.parse_args()
    args.train_split = args.split
    return args


if __name__ == '__main__':
    main()
