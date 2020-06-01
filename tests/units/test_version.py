import pytest

import eml.__version__ as __version__


def test_version_type():
    assert isinstance(__version__, str)
    assert __version__.count('.') == 2
