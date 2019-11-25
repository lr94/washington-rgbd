import washington

_rgb_mean = [0.5506, 0.5224, 0.5090]
_rgb_std = [0.2105, 0.2216, 0.2536]
_d_mean = [0.7438, 0.2960, 0.4763]
_d_std = [0.2013, 0.2026, 0.2904]


def init_washington_datasets(rgb_root=None, d_root=None, train_split=None, test_split=None, dataset_type='rgbd'):
    if train_split is None and test_split is None:
        raise ValueError("Invalid arguments: select at least a split to load")

    print("Loading dataset...")

    if dataset_type in ['rgb', 'd']:
        from torchvision.transforms import Normalize, RandomVerticalFlip, RandomHorizontalFlip, Resize, RandomCrop,\
            ToTensor, CenterCrop, Compose
    elif dataset_type == 'rgbd':
        from washington.transforms import Normalize, RandomVerticalFlip, RandomHorizontalFlip, Resize, RandomCrop,\
            ToTensor, CenterCrop, Compose, Concat
    else:
        raise ValueError("Invalid dataset type")

    normalize = None
    if dataset_type == 'rgb':
        normalize = Normalize(_rgb_mean, _rgb_std)
    elif dataset_type == 'd':
        normalize = Normalize(_d_mean, _d_std)
    elif dataset_type == 'rgbd':
        normalize = Normalize((_rgb_mean, _rgb_std), (_d_mean, _d_std))

    train_transform = Compose([
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        Resize(256),
        RandomCrop(224),

        ToTensor(),
        normalize
    ])

    test_transform = Compose([
        Resize(256),
        CenterCrop(224),

        ToTensor(),
        normalize
    ])

    train_set = None
    test_set = None
    if dataset_type in ['rgb', 'd']:
        root = rgb_root or d_root
        train_set = washington.WashingtonDataset3C(root, split=train_split, transform=train_transform,
                                                   dataset_type=dataset_type) if train_split is not None else None
        test_set = washington.WashingtonDataset3C(root, split=test_split, transform=test_transform,
                                                  dataset_type=dataset_type) if test_split is not None else None
    elif dataset_type == 'rgbd':
        train_transform.transforms.append(Concat())
        test_transform.transforms.append(Concat())
        train_set = washington.WashingtonDataset(rgb_root=rgb_root, d_root=d_root,
                                                 split=train_split, transform=train_transform) \
            if train_split is not None else None
        test_set = washington.WashingtonDataset(rgb_root=rgb_root, d_root=d_root,
                                                split=test_split, transform=test_transform) \
            if test_split is not None else None

    print("Dataset loaded")
    print("\tClasses: {}".format(len((train_set or test_set).class_labels)))

    if train_set is not None:
        print("\tTraining samples: {}".format(len(train_set)))
    if test_set is not None:
        print("\tTest samples: {}".format(len(test_set)))

    if (test_set is None) != (train_set is None):
        return train_set or test_set
    else:
        return train_set, test_set
