from torch.utils.data.dataset import Dataset
from ._utils import *

# TODO Rewrite documentation


class WashingtonDataset(Dataset):
    """
    Washington RGB-D <https://rgbd-dataset.cs.washington.edu>
    """

    def __init__(self, root, split, dataset_type='rgb', transform=None):
        self.root = root

        self.transform = transform

        self.class_labels = get_directories(root)
        self.class_ids_by_labels = {item: i for i, item in enumerate(self.class_labels)}

        suffix = None
        if dataset_type == 'rgb':
            suffix = 'crop.png'
        elif dataset_type == 'normal++':
            suffix = 'depthcrop.png'
        else:
            raise ValueError("Invalid argument type='{}'".format(dataset_type))

        with open(split, 'r') as f_split:
            self.file_paths = list(map(lambda line: line.strip().split(' ')[0] + suffix,
                                       f_split.readlines()))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, item):
        """
        :param item:                Index
        :return:                    (image, class) if only one of load_rgbd and load_depth is true.
                                        ((rgb_image, depth_image), class) if they are both true.
        """

        filepath = self.file_paths[item]

        label = filepath.split('/')[0]
        class_id = self.class_ids_by_labels[label]

        image = load_image(join(self.root, filepath))
        # Apply transform
        if self.transform is not None:
            image = self.transform(image)

        return image, class_id
