#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""
from setuptools import setup, find_namespace_packages
from pkg_resources import parse_requirements

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements/prod.txt') as prod_req:
    requirements = [str(ir) for ir in parse_requirements(prod_req)]
with open('requirements/test.txt') as test_req:
    test_requirements = [str(tr) for tr in parse_requirements(test_req)]

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
    tests_require= requirements + test_requirements,
    url='https://github.com/univieCUBE/phenotrex',
    version='0.4.0',
    zip_safe=False,
)
