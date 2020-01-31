#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

import codecs
from os import path
import re
from setuptools import setup, find_namespace_packages
from pip._internal.req import parse_requirements
from pip._internal.download import PipSession


# Single-sourcing the package version: Read from __init__
def read(*parts):
    here = path.abspath(path.dirname(__file__))
    with codecs.open(path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

parsed_requirements = parse_requirements('requirements/prod.txt', session=PipSession())
parsed_test_requirements = parse_requirements('requirements/test.txt', session=PipSession())

requirements = [str(ir.req) for ir in parsed_requirements]
test_requirements = [str(tr.req) for tr in parsed_test_requirements]

setup(
    author="Lukas Lüftinger",
    author_email='lukas.lueftinger@outlook.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
    ],
    description="Microbial Phenotype Prediction",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/x-rst',
    include_package_data=True,
    keywords='phenotrex',
    name='phenotrex',
    entry_points={'console_scripts': [
        'phenotrex = phenotrex.cli.main:main',
    ], },
    packages=find_namespace_packages(),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/univieCUBE/phenotrex',
    version=find_version("phenotrex", "__init__.py"),  # update there
    zip_safe=False,
)
