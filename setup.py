#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os

from setuptools import find_packages, setup

# Meta-data
NAME = "qmxplainer"
DESCRIPTION = "Explained Machine Learning - Library for interpretability methods"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
EMAIL = "mdb.sok@gmail.com"
AUTHOR = "Michaël SOK"
REQUIRES_PYTHON = ">=3.6.0"
VERSION = None

# Reading README.md
this_path = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(this_path, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

with io.open(os.path.join(this_path, "requirements.txt"), encoding="utf-8") as f:
    required_packages = f.read().split("\n")

# Imports version from module
about = {}
if not VERSION:
    with open(os.path.join(this_path, NAME, "__version__.py")) as f:
        exec(f.read(), about)
else:
    about["__version__"] = VERSION

# Setup
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url="",
    install_requires=required_packages,
    packages=find_packages(exclude=("tests",)),
    classifiers=[
        "License :: OSI Approved",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8"
    ]
)
