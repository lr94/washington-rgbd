#!/usr/bin/env python3
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
import torch.optim
import argparse

from net import *
from utils import *
from loader import init_washington_datasets


def main():
    args = parse_args()

    training_set, test_set = init_washington_datasets(args.dataset_root, train_split=args.training_split,
                                                      test_split=args.test_split, dataset_type=args.dataset_type)

    net = init_network(len(training_set.class_labels), pretrained=True)

    device, net, batch_size, last_epoch = init_device_model(args, net, model_file=args.model_path)

    tb_exp = args.tensorboard_exp
    if tb_exp is not None:
        tb_exp = os.path.join('runs', tb_exp)
    logger = Logger(use_tensorboard=args.tensorboard, tensorboard_logdir=tb_exp,
                    model_save_path=args.model_path, model=net, log_file=args.log_file,
                    input_shape=training_set[0][0].shape)

    train(net, training_set, test_set, args.epochs, batch_size=batch_size, lr=args.lr,
          momentum=args.momentum, device=device, logger=logger, initial_epoch=last_epoch + 1,
          lr_policy_milestones=args.lr_policy_milestones, lr_policy_gamma=args.lr_policy_gamma)


def train(model: nn.Module, training_set: Dataset, test_set: Dataset, epochs, batch_size=64, lr=0.01, momentum=0.0001,
          device=None, logger=None, initial_epoch=1, lr_policy_milestones=None, lr_policy_gamma=0.1):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)

    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    scheduler = None
    if lr_policy_milestones is not None:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=lr_policy_milestones, gamma=lr_policy_gamma)

    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

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
                       loss=loss.item(), batch_size=batch_size, accuracy=correct / total)

        correct, total, loss = test(model, test_loader, device=device, logger=logger)

        if scheduler is not None:
            scheduler.step(epoch=epoch)

        if logger is not None:
            logger.end_epoch(epoch, accuracy=correct / total, loss=loss)


def test(model: nn.Module, test_loader: DataLoader, device=None, logger=None):
    if device is not None:
        model = model.to(device)

    model.eval()

    correct = 0
    total = 0
    total_loss = 0.

    logger.pb = None

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

    dataset_opt_g = parser.add_argument_group(title="Dataset options")
    dataset_opt_g.add_argument('--dataset-root', default='./data', help="Folder containing the dataset")
    dataset_opt_g.add_argument('--dataset-type', default='rgb', choices=['rgb', 'normal++'])

    dataset_opt_g.add_argument('--training-split', help="Dataset split (.txt) to use")
    dataset_opt_g.add_argument('--test-split', help="Dataset split (.txt) to use for validiation")

    logging_opt_g = parser.add_argument_group(title="Logging options")
    logging_opt_g.add_argument('--tensorboard', action='store_true', help="Enable TensorBoard logging")
    logging_opt_g.add_argument('--tensorboard-exp', default=None, help="Tensorboard experiment name (logs will be "
                                                                       "saved in runs/TENSORBOARD_EXP)")
    logging_opt_g.add_argument('--log-file', default=None, help="Log file")

    add_device_options(parser)

    training_opt_g = parser.add_argument_group(title="Training options")
    training_opt_g.add_argument('--lr', type=float, default=0.01, help="Learning rate. Decrease it if "
                                                                       "increasing batch size")
    training_opt_g.add_argument('--lr-policy-milestones', type=int, nargs='+', default=None, help="Epoch indices")
    training_opt_g.add_argument('--lr-policy-gamma', type=float, default=0.1, help="Decay factor")
    training_opt_g.add_argument('--momentum', type=float, default=0, help="SGD Momentum")
    training_opt_g.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    training_opt_g.add_argument('--model-path', default=None, help="Path to load and store .pth model")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
