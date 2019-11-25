import warnings
import random
import numbers

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as t


class RGBDTransform:
    """
    Create a double transform starting from two "single" standard torchvision transforms
    """

    def __init__(self, both=None, rgb=None, depth=None):
        """
        None: Identity

        :param both:                Transform which has to be applied to both images. Can't be used with the other two.
                                        This is pretty much the same than using directly the "single" transform, but
                                        it does not generate the warning
        :param rgb:                 Transform which has to be applied to the rgb image
        :param depth:               Transform which has to be applied to the depth image
        """

        assert (rgb is not None or depth is not None) ^ (both is not None)

        if both is not None:
            self.rgb_transform = both
            self.d_transform = both
        else:
            self.rgb_transform = rgb
            self.d_transform = depth

    def __call__(self, rgb_image, d_image):
        rgb_image = rgb_image if self.rgb_transform is None else self.rgb_transform(rgb_image)
        d_image = d_image if self.d_transform is None else self.d_transform(d_image)

        return rgb_image, d_image


class Compose:
    """
    Compose multiple double or single transforms together forming a transform-chain.
    Single transforms are applied to both image streams.
    """

    def __init__(self, transforms):
        """
        :param transforms:          List of transforms
        """

        self.transforms = transforms

    def __call__(self, rgb_image, d_image):
        result = rgb_image, d_image

        for current_transform_i, current_transform in enumerate(self.transforms):
            result = current_transform(result[0], result[1])

        return result

    def __repr__(self):
        str_repr = self.__class__.__name__ + '('
        for current_t in self.transforms:
            str_repr += "\n\t{}".format(current_t)
        str_repr += "\n)"

        return str_repr


class RandomVerticalFlip:
    """
    Vertically flip two images randomly with the given probability
    """

    def __init__(self, p: float = 0.5):
        assert 0. <= p <= 1.
        self.p = p

    def __call__(self, rgb_image, d_image):
        if random.random() < self.p:
            return F.vflip(rgb_image), F.vflip(d_image)

        return rgb_image, d_image

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomHorizontalFlip:
    """
    Horizontally flip two PIL images randomly with the given probability
    """

    def __init__(self, p: float = 0.5):
        assert 0. <= p <= 1.
        self.p = p

    def __call__(self, rgb_image, d_image):
        if random.random() < self.p:
            return F.hflip(rgb_image), F.hflip(d_image)

        return rgb_image, d_image

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)


class RandomCrop(object):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def _get_params(self, image):
        target_height, target_width = self.size
        image_width, image_height = image.size
        if image_width == target_width and image_height == target_height:
            return 0, 0, image_height, image_width

        topleft_x = random.randint(0, image_height - target_height)
        topleft_y = random.randint(0, image_width - target_width)
        return topleft_x, topleft_y, target_height, target_width

    def __call__(self, rgb_image, d_image):
        if self.padding is not None:
            rgb_image = F.pad(rgb_image, self.padding, self.fill, self.padding_mode)
            d_image = F.pad(d_image, self.padding, self.fill, self.padding_mode)

        rgb_image = self._pad(rgb_image)
        d_image = self._pad(d_image)

        topleft_x, topleft_y, height, width = self._get_params(rgb_image)

        rgb_image = F.crop(rgb_image, topleft_x, topleft_y, height, width)
        d_image = F.crop(d_image, topleft_x, topleft_y, height, width)

        return rgb_image, d_image

    def _pad(self, img):
        width, height = img.size
        target_height, target_width = self.size

        # Pad width
        if self.pad_if_needed and width < target_width:
            img = F.pad(img, (target_width - width, 0), self.fill, self.padding_mode)
        # Pad height
        if self.pad_if_needed and height < target_height:
            img = F.pad(img, (0, target_height - height), self.fill, self.padding_mode)

        return img

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class Resize(RGBDTransform):

    def __init__(self, *args, **kwargs):
        super().__init__(both=t.Resize(*args, **kwargs))


class CenterCrop(RGBDTransform):

    def __init__(self, *args, **kwargs):
        super().__init__(both=t.CenterCrop(*args, **kwargs))


# Tensor transforms


class ToTensor(RGBDTransform):

    def __init__(self):
        super().__init__(both=t.ToTensor())


class Normalize(RGBDTransform):

    def __init__(self, means, stds):
        super().__init__(rgb=t.Normalize(means[0], stds[0]), depth=t.Normalize(means[1], stds[1]))


class Concat:

    """
    Concats two tensors CxHxW and DxHxW in a single tensor (C+D)xHxW
    """

    def __init__(self):
        pass

    def __call__(self, rgb_tensor: torch.Tensor, d_tensor: torch.Tensor):
        return torch.cat((rgb_tensor, d_tensor), dim=0)
