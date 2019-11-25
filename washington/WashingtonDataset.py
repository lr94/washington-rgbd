import warnings

from torch.utils.data.dataset import Dataset

from .WashingtonDataset3C import WashingtonDataset3C


class WashingtonDataset(Dataset):

    def __init__(self, rgb_root, d_root, split, transform=None):
        self.rgb_root = rgb_root
        self.d_root = d_root

        self.rgb_dataset = WashingtonDataset3C(rgb_root, split, dataset_type='rgb')
        self.d_dataset = WashingtonDataset3C(d_root, split, dataset_type='d')

        self.class_labels = self.rgb_dataset.class_labels
        self.class_ids_by_labels = self.rgb_dataset.class_ids_by_labels

    def __len__(self):
        dataset_size = len(self.rgb_dataset)
        assert dataset_size == len(self.d_dataset)

        return dataset_size

    def __getitem__(self, item):
        rgb_image, rgb_class_id = self.rgb_dataset[item]
        d_image, d_class_id = self.d_dataset[item]

        assert rgb_class_id == d_class_id

        result = rgb_image, d_image
        if self._transform is not None:
            result = self._transform(rgb_image, d_image)

        return result, rgb_class_id
