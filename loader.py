import time
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
        # Computed on split "train cut 0"
        tr.transforms.append()

    return tr


def init_washington_datasets(dataset_root, train_split, test_split=None):
    normalize_transform = torchvision.transforms.Normalize([0.5506, 0.5224, 0.5090], [0.2105, 0.2216, 0.2536])

    training_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip(),

        torchvision.transforms.Resize(256),
        torchvision.transforms.RandomCrop(224),
        torchvision.transforms.ToTensor(),
        normalize_transform
    ])

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        normalize_transform
    ])

    start_time = time.time()
    print("Loading dataset...")
    train_set = washington.WashingtonDataset(dataset_root, split=train_split, transform=training_transform)
    test_set = None
    if test_split is not None:
        test_set = washington.WashingtonDataset(dataset_root, split=test_split, transform=test_transform)
    print("Dataset loaded")
    print("\tSamples: {}".format(len(train_set)))
    print("\tClasses: {}".format(len(train_set.class_labels)))

    print("\tTraining samples: {}".format(len(train_set)))
    if test_split is not None:
        print("\tTest samples: {}".format(len(test_set)))

    end_time = time.time()
    print('Dataset loaded in {:.3f} s'.format(end_time - start_time))

    if test_set is None:
        return train_set
    else:
        return train_set, test_set
