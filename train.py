#!/usr/bin/env python3
import time
import torch
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as f
import torch.optim
import torchvision.transforms
import torchvision.models
import washington
import argparse
from utils import *


def main():
    parser = argparse.ArgumentParser(
        description="Sample program for the Washington RGB-D Dataset. The program trains a ResNet-18 using only RGB "
                    "images"
    )
    parser.add_argument('--dataset-root', default='./data', help="Folder containing the dataset (it must contain the"
                                                                 "directory \"rgb-dataset\")")
    parser.add_argument('--training-split', type=float, default=0.9, help="Fraction of the dataset to be used as "
                                                                          "training set")
    parser.add_argument('--logging-period', type=int, default=1, help="Number of batches between log updates")
    parser.add_argument('--tensorboard', action='store_true', help="Enable TensorBoard logging")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size for training, testing and mean/std "
                                                                   "computation")
    parser.add_argument('--learning-rate', type=float, default=0.01, help="Learning rate. Decrease it if increasing "
                                                                          "batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--momentum', type=float, default=0.0001, help="SGD Momentum")
    parser.add_argument('--nesterov', action='store_true', help="Enable Nesterov SGD")
    parser.add_argument('--model-path', default=None, help="Path to load and store .pt model")

    parser.add_argument('--compute-stats', action='store_true', help="Compute mean and standard deviation of the "
                                                                     "whole dataset")

    args = parser.parse_args()

    if args.compute_stats:
        stats(dataset_path=args.dataset_root, batch_size=args.batch_size)
    else:
        train_job(dataset_path=args.dataset_root, training_split=args.training_split,
                  logging_period=args.logging_period, use_tensorboard=args.tensorboard, batch_size=args.batch_size,
                  lr=args.learning_rate, momentum=args.momentum, nesterov=args.nesterov, epochs=args.epochs,
                  model_path=args.model_path)


# I'm using a function with a lot of keyword arguments instead of passing directly "parser.parse_args()" because I
# want to be able to easily call this function from a Jupyter notebook without passing by the entry point main()
def train_job(dataset_path=None, training_split=0.9, logging_period=1, use_tensorboard=False, batch_size=1, lr=0.01,
              momentum=0.0001, nesterov=True, device=None, epochs=10, model_path=None):
    tr = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5438, 0.5168, 0.5042], [0.2119, 0.2205, 0.2511])
    ])

    start_time = time.time()
    print("Loading dataset...")
    full_dataset = washington.WashingtonDataset(dataset_path, download=True, transform=tr)
    print("Dataset loaded and normalized")
    print("\tSamples: {}".format(len(full_dataset)))
    print("\tClasses: {}".format(len(full_dataset.class_labels)))

    training_size = int(training_split * len(full_dataset))
    training_set, test_set = random_split(full_dataset, [training_size, len(full_dataset) - training_size])
    print("\tTraining samples: {}".format(len(training_set)))
    print("\tTest samples: {}".format(len(test_set)))

    end_time = time.time()
    print('Dataset loaded in {:.3f} s'.format(end_time - start_time))

    logger = Logger(period=logging_period, use_tensorboard=use_tensorboard, model_save_path=model_path)

    # TODO fix: what if device is explicitly set to torch.device('cpu') ?
    device = device if device is not None else (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    if torch.cuda.is_available():
        print("GPU: {}".format(torch.cuda.get_device_name(device)))

    resnet = torchvision.models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(full_dataset.class_labels))

    if model_path is not None and os.path.exists(model_path):
        print("Loading model from {}".format(model_path))
        resnet.load_state_dic(torch.load(model_path))
        print("Model loaded")

    train(resnet, training_set, test_set, epochs, batch_size=batch_size, lr=lr, momentum=momentum, nesterov=nesterov,
          device=device, logger=logger)


def train(model: nn.Module, training_set: Dataset, test_set: Dataset, epochs, batch_size=64, lr=0.01, momentum=0.0001,
          nesterov=True, device=None, logger=None):
    if device is None:
        device = torch.device('cpu')
    model = model.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    test(model, test_loader, batch_size=batch_size, device=device, logger=logger)

    for epoch in range(1, epochs + 1):
        if logger is not None:
            logger.start_epoch(epoch, epochs)

        model.train()

        if logger is not None:
            logger.reset_epoch()
        for batch_i, (input_t, target_t) in enumerate(training_loader):
            input_t = input_t.to(device)
            target_t = target_t.to(device)

            opt.zero_grad()
            output_t = model(input_t)
            loss = f.cross_entropy(output_t, target_t)
            loss.backward()
            opt.step()

            if logger is not None:
                logger(epoch=epoch, batch_index=batch_i, samples=len(training_set), batches=len(training_loader),
                       loss=loss.item(), batch_size=batch_size)

        test(model, test_loader, batch_size=batch_size, device=device, logger=logger)

        if logger is not None:
            logger.end_epoch(epoch)


@measure
def test(model: nn.Module, test_loader: DataLoader, batch_size=64, device=None, logger=None):
    if device is not None:
        model = model.to(device)

    model.eval()

    correct = 0
    total = 0
    total_loss = 0.

    with torch.no_grad():
        for batch_i, (input_t, target_t) in enumerate(test_loader):
            input_t = input_t.to(device)
            target_t = target_t.to(device)

            output_t = model(input_t)

            total_loss += f.cross_entropy(output_t, target_t).item()

            predictions = torch.argmax(output_t, dim=1)
            correct += torch.sum(predictions == target_t).item()
            total += predictions.shape[0]

            if logger is not None:
                logger.log_test_progress(total, len(test_loader.dataset))

    loss = total_loss / len(test_loader)

    if logger is not None:
        logger.log_test_result(correct, total, loss)

    return correct, total, loss


def stats(dataset_path=None, batch_size=32):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.ToTensor(),
    ])

    start_time = time.time()
    print("Loading dataset...")
    full_dataset = washington.WashingtonDataset(dataset_path, download=True, transform=transforms)
    print("Dataset loaded")
    print("\tSamples: {}".format(len(full_dataset)))
    print("\tClasses: {}".format(len(full_dataset.class_labels)))
    end_time = time.time()
    print('Dataset loaded in {:.3f} s'.format(end_time - start_time))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result = compute_avg_std(full_dataset, batch_size=batch_size, device=device)

    print("Mean: ", result[0])
    print("Std:  ", result[1])


@measure
def compute_avg_std(dataset, batch_size=16, device=None):
    """
    Compute dataset average and standard deviation using an online algorithm (Welford, 1962)
    Both statistics are computed for each channel
    :param dataset:
    :param batch_size:
    :param device: None, torch.device('cpu') or torch.device('cuda')
    :return: A tuple containing mean and standard deviation
    """
    from torch.utils.data.dataloader import DataLoader
    from torch.hub import tqdm

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


if __name__ == '__main__':
    main()
