import argparse
import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
import torchvision

import washington
from utils import *


def main():
    args = parse_args()

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.ToTensor(),
    ])

    print("Loading dataset...")
    full_dataset = washington.WashingtonDataset(args.dataset_root, download=True, transform=transforms)
    print("Dataset loaded")
    print("\tSamples: {}".format(len(full_dataset)))

    device = get_device(enable_cuda=not args.disable_cuda, cuda_device_id=args.cuda_device)

    result = compute_avg_std(full_dataset, batch_size=args.batch_size, device=device)

    print("Mean: ", result[0])
    print("Std:  ", result[1])


@stopwatch
def compute_avg_std(dataset, batch_size=64, device=None):
    """
    Compute dataset average and standard deviation using an online algorithm (Welford, 1962)
    Both statistics are computed for each channel
    :param dataset:
    :param batch_size:
    :param device: None, torch.device('cpu') or torch.device('cuda')
    :return: A tuple containing mean and standard deviation
    """

    loader = DataLoader(dataset, batch_size)

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

    parser.add_argument('--batch-size', type=int, default=64, help="Batch size for mean/std "
                                                                   "computation")

    training_opt_g = parser.add_argument_group(title="Dataset options")
    training_opt_g.add_argument('--dataset-root', default='./data', help="Folder containing the dataset (it must "
                                                                         "contain the directory \"rgb-dataset\")")

    device_opt_g = parser.add_argument_group(title="Device options")
    device_opt_g.add_argument('--disable-cuda', action='store_true', help="Disable GPU acceleration")
    device_opt_g.add_argument('--cuda-device', type=int, help="Select a specific GPU")

    return parser.parse_args()


if __name__ == '__main__':
    main()
