import time
import torch
from torch.utils.data import random_split, Dataset, DataLoader
import torchvision.transforms

import washington


def _init_transforms(normalize=True):
    tr = torchvision.transforms.Compose([
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),

        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.ToTensor()
    ])

    if normalize:
        tr.transforms.append(torchvision.transforms.Normalize([0.5438, 0.5168, 0.5042], [0.2119, 0.2205, 0.2511]))

    return tr


def init_washington_datasets(dataset_root, testset_root=None, training_split=None, normalize=True):
    """
    Prepare two datasets (training set and test set). If testset_root is provided the two datasets are the ones
    present in dataset_root and testset_root, otherwise training_split must be specified and the dataset contained in
    dataset_root is randomly splitted
    :param dataset_root:
    :param testset_root:
    :param training_split:
    :param normalize:
    :return:
    """

    if testset_root is not None and training_split is not None:
        raise ValueError("You can't specify both testset_root and training_split")

    tr = _init_transforms(normalize)

    start_time = time.time()
    print("Loading dataset...")
    full_dataset = washington.WashingtonDataset(dataset_root, download=True, transform=tr)
    print("Dataset loaded and normalized")
    print("\tSamples: {}".format(len(full_dataset)))
    print("\tClasses: {}".format(len(full_dataset.class_labels)))

    if training_split is not None:
        print("Randomly splitting dataset...")
        training_size = int(training_split * len(full_dataset))
        training_set, test_set = random_split(full_dataset, [training_size, len(full_dataset) - training_size])
        training_set.classes = full_dataset.classes
        test_set.classes = full_dataset.classes
        training_set.class_labels = full_dataset.class_labels
        test_set.class_labels = full_dataset.class_labels
        training_set.root = full_dataset.root
        test_set.root = full_dataset.root
    elif testset_root is not None:
        print("Loading training set...")
        training_set = full_dataset
        test_set = washington.WashingtonDataset(testset_root, download=False, transform=tr)
    else:
        training_set = full_dataset
        test_set = None

    print("\tTraining samples: {}".format(len(training_set)))

    if test_set is not None:
        print("\tTest samples: {}".format(len(test_set)))

    end_time = time.time()
    print('Dataset loaded in {:.3f} s'.format(end_time - start_time))

    return training_set, test_set
