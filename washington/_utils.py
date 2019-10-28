from os.path import isdir, isfile, join
from os import listdir
from re import match
import PIL.Image


def get_directories(path: str, pattern: str = None):
    """
    Given a directory, list all the subdirectories matching a given pattern
    :param path:            Path of the root directory
    :param pattern:         Pattern
    :return:                A list of the subdirectories
    """

    return list(filter(lambda n: isdir(join(path, n)) and (pattern is None or bool(match(pattern, n))), listdir(path)))


def get_files(path: str, pattern: str = None):
    """
    Given a directory, list all the contained files matching a given pattern
    :param path:            Path of the root directory
    :param pattern:         Pattern
    :return:                File list
    """

    return list(filter(lambda n: isfile(join(path, n)) and (pattern is None or bool(match(pattern, n))), listdir(path)))


def load_image(path: str):
    with open(path, 'rb') as fh:
        return PIL.Image.open(fh).convert('RGB')


def count_callable_args(callable_object):
    """
    Count the number of arguments of a callable object
    :param callable_object: Object
    :return:                Number of arguments
    """

    from inspect import getfullargspec, ismethod, isroutine

    # Actual number of arguments
    n = len(getfullargspec(callable_object).args)
    # If the object is a method or a common object implementing __call__ don't count "self"
    n = n - 1 if ismethod(callable_object) or not isroutine(callable_object) else n
    # Non-static method without "self" argument
    if n < 0:
        raise TypeError()
    return n
