#!/usr/bin/env python3
import argparse

from torch.utils.data import DataLoader

from net import *
from utils import Logger, add_device_options, init_device_model
from loader import init_washington_datasets
from train import test


def main():
    args = parse_args()

    test_set = init_washington_datasets(args.dataset_root, test_split=args.test_split)

    net = init_network(len(test_set.class_labels), pretrained=False)

    device, net, batch_size, _ = init_device_model(args, net, model_file=args.model_path)

    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    logger = Logger()
    test(net, test_loader, device, logger)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a model",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100)
    )

    dataset_opt_g = parser.add_argument_group(title="Dataset options")
    dataset_opt_g.add_argument('--dataset-root', default='./data', help="Folder containing the dataset")
    dataset_opt_g.add_argument('--test-split', help="Dataset split (.txt) to use for validiation")

    add_device_options(parser)

    model_opt_g = parser.add_argument_group(title="Model options")
    model_opt_g.add_argument('--model-path', default=None, help="Path to load .pth model")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
