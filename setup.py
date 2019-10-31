#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_namespace_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

# Requirements are required here, while requirements.txt is likely not required.
requirements = ['numpy', 'scipy', 'matplotlib', 'scikit-learn', 'xgboost', 'click']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

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
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/univieCUBE/PICA2',
    version='0.2.0',
    zip_safe=False,
)
