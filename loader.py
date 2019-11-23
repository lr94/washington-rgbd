import torchvision.transforms

import washington


def init_washington_datasets(dataset_root, train_split=None, test_split=None, dataset_type='rgb'):
    if train_split is None and test_split is None:
        raise ValueError("Invalid arguments: select at least a split to load")

    dataset_mean = None
    dataset_std = None
    if dataset_type == 'rgb':
        dataset_mean = [0.5506, 0.5224, 0.5090]
        dataset_std = [0.2105, 0.2216, 0.2536]
    elif dataset_type == 'normal++':
        dataset_mean = [0.7438, 0.2960, 0.4763]
        dataset_std = [0.2013, 0.2026, 0.2904]

    normalize_transform = torchvision.transforms.Normalize(dataset_mean, dataset_std)

    print("Loading dataset...")

    train_set = None
    if train_split is not None:
        training_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomHorizontalFlip(),

            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(224),
            torchvision.transforms.ToTensor(),
            normalize_transform
        ])
        train_set = washington.WashingtonDataset(dataset_root, split=train_split, transform=training_transform,
                                                 dataset_type=dataset_type)

    test_set = None
    if test_split is not None:
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            normalize_transform
        ])
        test_set = washington.WashingtonDataset(dataset_root, split=test_split, transform=test_transform,
                                                dataset_type=dataset_type)

    print("Dataset loaded")
    print("\tClasses: {}".format(len((train_set if train_set is not None else test_set).class_labels)))

    if train_set is not None:
        print("\tTraining samples: {}".format(len(train_set)))
    if test_set is not None:
        print("\tTest samples: {}".format(len(test_set)))

    if test_set is None and train_set is not None:
        return train_set
    elif test_set is not None and train_set is None:
        return test_set
    else:
        return train_set, test_set
