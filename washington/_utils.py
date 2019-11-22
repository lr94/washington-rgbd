from os.path import isdir, join
from os import listdir
import PIL.Image


def get_directories(path: str):
    """
    Given a directory, list all the subdirectories matching a given pattern
    :param path:            Path of the root directory
    :return:                A list of the subdirectories
    """

    return list(filter(lambda n: isdir(join(path, n)), listdir(path)))


def load_image(path: str):
    with open(path, 'rb') as fh:
        return PIL.Image.open(fh).convert('RGB')
