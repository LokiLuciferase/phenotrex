#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_namespace_packages
from pip._internal.req import parse_requirements
from pip._internal.download import PipSession


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
    version='0.4.0',
    zip_safe=False,
)
