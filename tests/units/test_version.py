import pytest

from eml import __version__


MODULE_ATTRIBUTES = [
    '__name__',
    '__doc__',
    '__package__',
    '__loader__',
    '__spec__',
    '__file__',
    '__cached__',
    '__builtins__'
]


def test_constants_in_version():
    constants = ('VERSION', '__version__')
    for c in __version__.__dict__.keys():
        assert (c in MODULE_ATTRIBUTES) or (c in constants)


def test_version_type():
    assert isinstance(__version__.VERSION, tuple)
    assert len(__version__.VERSION) == 3
    for element in __version__.VERSION:
        assert isinstance(element, (int, str))

    assert isinstance(__version__.__version__, str)
    assert __version__.__version__.count('.') == 2
