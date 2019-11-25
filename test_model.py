#!/usr/bin/env python3
import argparse

from torch.utils.data import DataLoader

from net import *
from utils import Logger, add_device_options, parse_device_model_args, parse_dataset_args
from loader import init_washington_datasets
from train import test


def main():
    args = parse_args()

    test_set = parse_dataset_args(args)

    net = init_network(len(test_set.class_labels), pretrained=False, input_channels=test_set[0][0].shape[0])

    device, workers, net, batch_size, _ = parse_device_model_args(args, net, model_file=args.model_path)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=workers)

    logger = Logger()
    test(net, test_loader, device, logger)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a model",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100)
    )

    dataset_opt_g = parser.add_argument_group(title="Dataset options")
    dataset_opt_g.add_argument('--rgb-root', default=None, help="Folder containing the RGB dataset")
    dataset_opt_g.add_argument('--d-root', default=None, help="Folder containing the Depth dataset")
    dataset_opt_g.add_argument('--test-split', help="Dataset split (.txt) to use for validiation")

    add_device_options(parser)

    model_opt_g = parser.add_argument_group(title="Model options")
    model_opt_g.add_argument('--model-path', default=None, help="Path to load .pth model")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
