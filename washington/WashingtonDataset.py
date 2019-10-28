import warnings
import os
import re
from torch.utils.data.dataset import Dataset
from torchvision.datasets.utils import download_and_extract_archive
from ._utils import *


class WashingtonDataset(Dataset):
    """
    Washington RGB-D <https://rgbd-dataset.cs.washington.edu> Dataset
    At least one of load_rgb and load_depth must be True
    """

    url = 'https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset/rgbd-dataset.tar'
    _dirname = 'rgbd-dataset'

    def __init__(self, root, load_rgb=True, load_depth=False, transform=None, target_transform=None, download=False):
        """
        :param root:                Root of the dataset (containing the "rgbd-dataset" directory)
        :param load_rgb:            Specify whether RGB images have to be loaded
        :param load_depth:          Specify whether Depth images have to be loaded
        :param transform:           Transforms
        :param target_transform:    Target transforms
        :param download:            If true the dataset is downloaded and extracted in the root directory (if it hasn't
                                        been downloaded yet)
        """

        self.root = root
        self._transform = transform
        self._target_transform = target_transform

        if transform is not None:
            # How many arguments does the selected transform take?
            self._transform_nargs = count_callable_args(transform)
            # How many arguments do we expect?
            needed_args = load_rgb + load_depth
            # If the given transform accepts only one argument and we need more, let's warn the user but proceed anyway
            if self._transform_nargs < needed_args and self._transform_nargs == 1:
                warnings.warn("Selected transform expects {0} arguments instead of {1}, the transform will be "
                              "applied independently on each image of a sample. Pay particular attention to "
                              "random transforms.".format(self._transform_nargs, needed_args), RuntimeWarning)
            # Otherwise if the expected and actual number of args don't match raise an exception
            elif self._transform_nargs != needed_args:
                raise TypeError("Selected transform expects {0} arguments instead of {1}."
                                .format(self._transform_nargs, needed_args))

        # If necessary download the dataset
        if not os.path.exists(root):
            if download:
                self.download()
            else:
                raise RuntimeError("Use download=True to download the dataset")

        # RGB? D? RGB-D?
        if not (load_rgb or load_depth):
            raise RuntimeError("Select at least one between RGB and Depth")
        self._load_rgb = load_rgb
        self._load_depth = load_depth

        self.class_labels = get_directories(join(root, self._dirname))

        # Base for file names ("rgbd-dataset/notebook/notebook_1/notebook_1_2_215" for "notebook_1_2_215_crop.png")
        self._base_file_names = []
        self.classes = []

        # Categories
        for current_class_id, current_class_label in enumerate(self.class_labels):
            class_path = os.path.join(root, self._dirname, current_class_label)

            instances = get_directories(class_path)

            # Instances
            for current_instance in instances:
                instance_path = os.path.join(class_path, current_instance)

                # Frames (samples)
                frames = get_files(instance_path, pattern=current_instance + r'_\d+_\d+_crop\.png')
                base_frame_names = map(lambda n: re.match(r'([\w\d_]+)_crop\.png', n).group(1), frames)

                # "rgbd-dataset/notebook/notebook_1/notebook_1_2_215"
                self._base_file_names.extend(map(lambda n: join(current_class_label, current_instance, n),
                                                 base_frame_names))
                self.classes.extend([current_class_id] * len(frames))

    def __len__(self):
        return len(self._base_file_names)

    def __getitem__(self, item):
        """
        :param item:                Index
        :return:                    (image, class) if only one of load_rgbd and load_depth is true.
                                        ((rgb_image, depth_image), class) if they are both true.
        """

        image_class = self.classes[item]

        rgb_image = None
        d_image = None

        if self._load_rgb:
            rgb_image_path = join(self.root, self._dirname, self._base_file_names[item]) + '_crop.png'
            rgb_image = load_image(rgb_image_path)

        if self._load_depth:
            d_image_path = join(self.root, self._dirname, self._base_file_names[item]) + '_depthcrop.png'
            d_image = load_image(d_image_path)

        # Value to return
        imgs = None

        # Ensure we have something to load
        assert self._load_rgb or self._load_depth

        # Load only RGB
        if self._load_rgb and not self._load_depth:
            # Apply transform
            if self._transform is not None:
                rgb_image = self._transform(rgb_image)
            imgs = rgb_image
        # Load only D
        elif self._load_depth and not self._load_rgb:
            # Apply transform
            if self._transform is not None:
                d_image = self._transform(d_image)
            imgs = d_image
        # Load both
        elif self._load_rgb and self._load_depth:
            # Apply transform:
            if self._transform is not None:
                # We need 2 arguments, if it gets only one let's apply the transform twice
                if self._transform_nargs == 1:
                    rgb_image, d_image = self._transform(rgb_image), self._transform(d_image)
                else:
                    rgb_image, d_image = self._transform(rgb_image, d_image)
            imgs = (rgb_image, d_image)

        # Apply target transform
        if self._target_transform is not None:
            image_class = self._target_transform(image_class)

        return imgs, image_class

    def download(self):
        download_and_extract_archive(self.url, os.path.curdir, self.root, remove_finished=True)
