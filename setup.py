#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from setuptools import setup, find_packages

try:
    README = open('README.md').read()
except Exception:
    README = ""
VERSION = "0.0.1"

requirments = ["modelhub"]

setup(
    name='code_practice',
    version=VERSION,
    description='code_practice',
    url="http://git.patsnap.com/research/code_practice",
    long_description=README,
    author='jianan',
    author_email='hujianan@patsnap.com',
    packages=find_packages(),
    install_requires=requirments,
    extras_require={
        # "extra": ["extra_requirments"],
    },
    entry_points={
        # 'console_scripts': [
        #     'modelhub=modelhub.commands:main'
        # ]
    },
)