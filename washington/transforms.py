import warnings
import random
import torchvision.transforms.functional as f
from ._utils import count_callable_args


class DoubleTransform:
    """
    Create a double transform starting from two "single" standard torchvision transforms
    """

    def __init__(self, rgb_transform=None, d_transform=None, both=None):
        """
        None: Identity
        :param rgb_transform:       Transform which has to be applied to the rgb image
        :param d_transform:         Transform which has to be applied to the depth image
        :param both:                Transform which has to be applied to both images. Can't be used with the other two.
                                        This is pretty much the same than using directly the "single" transform, but
                                        it does not generate the warning
        """

        assert (rgb_transform is not None or d_transform is not None) ^ (both is not None)

        if both is not None:
            self.rgb_transform = both
            self.d_transform = both
        else:
            self.rgb_transform = rgb_transform
            self.d_transform = d_transform

    def __call__(self, rgb_image, d_image):
        rgb_image = rgb_image if self.rgb_transform is None else self.rgb_transform(rgb_image)
        d_image = d_image if self.d_transform is None else self.d_transform(d_image)

        return rgb_image, d_image


class DoubleCompose:
    """
    Compose multiple double or single transforms together forming a transform-chain.
    Single transforms are applied to both image streams.
    """

    def __init__(self, transforms):
        """
        :param transforms:          List of transforms
        """

        self.transforms = transforms

        # Check number of arguments
        self._transforms_nargs = []
        for current_transform in transforms:
            n = count_callable_args(current_transform)

            # If the given transform accepts only one argument (we need two), let's warn the user but proceed anyway
            if n == 1:
                warnings.warn("Selected transform expects {0} arguments instead of {1}, the transform will be "
                              "applied independently on each image of a sample. Pay particular attention to "
                              "random transforms.".format(n, 2), RuntimeWarning)
            # Otherwise if the expected and actual number of args don't match raise an exception
            elif n != 2:
                raise TypeError("Selected transform expects {0} arguments instead of {1}."
                                .format(n, 2))

            self._transforms_nargs.append(n)

    def __call__(self, rgb_image, d_image):
        for current_transform_i, current_transform in enumerate(self.transforms):
            n = self._transforms_nargs[current_transform_i]
            if n == 1:
                rgb_image, d_image = current_transform(rgb_image), current_transform(d_image)
            elif n == 2:
                rgb_image, d_image = current_transform(rgb_image, d_image)

        return rgb_image, d_image

    def __repr__(self):
        str_repr = self.__class__.__name__ + '('
        for current_t in self.transforms:
            str_repr += "\n\t{}".format(current_t)
        str_repr += "\n)"

        return str_repr


class DoubleRandomVerticalFlip:
    """
    Vertically flip two PIL images randombly with the specified probability
    """

    def __init__(self, p: float = 0.5):
        """
        :param p:                   Probability
        """

        assert 0. <= p <= 1.
        self.p = p

    def __call__(self, rgb_image, d_image):
        if random.random() < self.p:
            return f.vflip(rgb_image), f.vflip(d_image)

        return rgb_image, d_image

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)
