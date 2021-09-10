
phenotrex
=========


.. image:: https://badge.fury.io/py/phenotrex.svg
   :target: https://pypi.python.org/pypi/phenotrex
   :alt: PyPI

.. image:: https://codecov.io/gh/univieCUBE/phenotrex/branch/master/graph/badge.svg
   :target: https://codecov.io/gh/univieCUBE/phenotrex
   :alt: Codecov

.. image:: https://img.shields.io/lgtm/grade/python/g/LokiLuciferase/phenotrex.svg?logo=lgtm&logoWidth=18
   :target: https://lgtm.com/projects/g/LokiLuciferase/phenotrex/context:python
   :alt: Code Quality

.. image:: https://travis-ci.com/univieCUBE/phenotrex.svg?branch=master
   :target: https://travis-ci.com/univieCUBE/phenotrex
   :alt: Travis CI

.. image:: https://ci.appveyor.com/api/projects/status/iursmhw1wocfgpua?svg=true
   :target: https://ci.appveyor.com/project/VarIr/phenotrex
   :alt: AppVeyor CI

.. image:: https://readthedocs.org/projects/phenotrex/badge/?version=latest
   :target: https://phenotrex.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status


End-to-end Microbial Phenotypic Trait Prediction.

Installation
------------

.. code-block::

    $ pip install phenotrex[fasta]

Usage
-----

Phenotrex is a component of the `PhenDB`_ web server, which performs phenotypic trait prediction on
user-uploaded metagenomic bins. To try out phenotrex with PhenDB's pre-trained and curated set of
trait models, genomes may thus simply be `submitted to PhenDB`_.

Basic Usage
~~~~~~~~~~~
To use a trained phenotrex model ``MY_TRAIT.pkl`` for prediction of a phenotypic trait with a
given genome ``genome.fna``:

.. code-block::

    $ phenotrex predict --classifier MY_TRAIT.pkl genome.fna > predictions.tsv


This yields a tabular file containing a prediction regarding the presence of the trait (YES or NO),
as well as a confidence value the model ascribes to this prediction, ranging from 0.5 to 1.0.

Advanced Usage
~~~~~~~~~~~~~~
For training, evaluation and explanation of phenotrex models on user data, please refer to the
full usage tutorial `here`_.

.. _PhenDB: https://www.phendb.org/
.. _submitted to PhenDB: https://phen.csb.univie.ac.at/phendb/
.. _here: https://phenotrex.readthedocs.io/en/latest/usage.html
