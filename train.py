#!/usr/bin/env python3
import torch
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as f
import torch.optim
import torchvision.models
import argparse

from utils import *
from loader import init_washington_datasets


def main():
    args = parse_args()

    dataset_path = args.dataset_root
    training_split = args.training_split
    batch_size = args.batch_size

    lr = args.learning_rate
    momentum = args.momentum
    nesterov = args.nesterov
    epochs = args.epochs

    training_set, test_set = init_washington_datasets(dataset_path, training_split=training_split,
                                                      normalize=True)

    device = get_device(enable_cuda=not args.disable_cuda, cuda_device_id=args.cuda_device)

    resnet = torchvision.models.resnet18(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(training_set.class_labels))

    tb_exp = args.tensorboard_exp
    if tb_exp is not None:
        tb_exp = os.path.join('runs', tb_exp)
    logger = Logger(period=args.logging_period, use_tensorboard=args.tensorboard, tensorboard_logdir=tb_exp,
                    model_save_path=args.model_path, model=resnet, log_file=args.log_file,
                    input_shape=training_set[0][0].shape)

    checkpoint_epoch = 0
    if args.model_path is not None and os.path.exists(args.model_path):
        print("Loading model from {}".format(args.model_path))
        data = torch.load(args.model_path, map_location=device)
        m_sd = data['m_state_dict']
        checkpoint_epoch = data['epoch']
        resnet.load_state_dict(m_sd)

        print("Model loaded")

    train(resnet, training_set, test_set, epochs, batch_size=batch_size, lr=lr, momentum=momentum, nesterov=nesterov,
          device=device, logger=logger, initial_epoch=checkpoint_epoch + 1, initial_test=not args.skip_initial_test)


def train(model: nn.Module, training_set: Dataset, test_set: Dataset, epochs, batch_size=64, lr=0.01, momentum=0.0001,
          nesterov=True, device=None, logger=None, initial_epoch=1, initial_test=True):
    if device is None:
        device = torch.device('cpu')
    model = model.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov)

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    if initial_test:
        test(model, test_loader, device=device, logger=logger)

    for epoch in range(initial_epoch, epochs + 1):
        if logger is not None:
            logger.start_epoch(epoch, epochs)

        model.train()

        for batch_i, (input_t, target_t) in enumerate(training_loader):
            input_t = input_t.to(device)
            target_t = target_t.to(device)

            opt.zero_grad()
            output_t = model(input_t)
            loss = f.cross_entropy(output_t, target_t)
            loss.backward()
            opt.step()

            predictions = torch.argmax(output_t, dim=1)
            correct = torch.sum(predictions == target_t).item()
            total = predictions.shape[0]

            if logger is not None:
                logger(epoch=epoch, batch_index=batch_i, samples=len(training_set), batches=len(training_loader),
                       loss=loss.item(), batch_size=batch_size, accuracy=correct/total)

        test(model, test_loader, device=device, logger=logger)

        if logger is not None:
            logger.end_epoch(epoch)


@stopwatch
def test(model: nn.Module, test_loader: DataLoader, device=None, logger=None):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sample program for the Washington RGB-D Dataset. The program trains a ResNet-18 using only RGB "
                    "images",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=50, width=100)
    )

    training_opt_g = parser.add_argument_group(title="Dataset options")
    training_opt_g.add_argument('--dataset-root', default='./data', help="Folder containing the dataset (it must "
                                                                         "contain the directory \"rgb-dataset\")")
    training_opt_g.add_argument('--training-split', type=float, default=0.9, help="Fraction of the dataset to be used "
                                                                                  "as training set")

    logging_opt_g = parser.add_argument_group(title="Logging options")
    logging_opt_g.add_argument('--logging-period', type=int, default=1, help="Number of batches between log updates")
    logging_opt_g.add_argument('--tensorboard', action='store_true', help="Enable TensorBoard logging")
    logging_opt_g.add_argument('--tensorboard-exp', default=None, help="Tensorboard experiment name (logs will be "
                                                                       "saved in runs/TENSORBOARD_EXP)")
    logging_opt_g.add_argument('--log-file', default=None, help="Log file")

    device_opt_g = parser.add_argument_group(title="Device options")
    device_opt_g.add_argument('--disable-cuda', action='store_true', help="Disable GPU acceleration")
    device_opt_g.add_argument('--cuda-device', type=int, help="Select a specific GPU")

    training_opt_g = parser.add_argument_group(title="Training options")
    training_opt_g.add_argument('--batch-size', type=int, default=64, help="Batch size for training and testing")
    training_opt_g.add_argument('--learning-rate', type=float, default=0.01, help="Learning rate. Decrease it if "
                                                                                  "increasing batch size")
    training_opt_g.add_argument('--skip-initial-test', action='store_true', help="Skip test before training")
    training_opt_g.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    training_opt_g.add_argument('--momentum', type=float, default=0.0001, help="SGD Momentum")
    training_opt_g.add_argument('--nesterov', action='store_true', help="Enable Nesterov SGD")
    training_opt_g.add_argument('--model-path', default=None, help="Path to load and store .pth model")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
