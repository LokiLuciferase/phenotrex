.. highlight:: shell

============
Installation
============


Stable release
--------------

For a full installation of phenotrex, run this command in your terminal:

.. code-block:: console

    $ pip install phenotrex[fasta]

Note that this command installs large dependencies (`pytorch`_, `deepnog`_) required for
transforming FASTA files into phenotrex input features at runtime.

If this capability is not needed (for example because feature files
have been pre-created),
installation size can be significantly reduced by running instead:

.. code-block:: console

    $ pip install phenotrex

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pytorch: https://pytorch.org/
.. _deepnog: https://github.com/univieCUBE/deepnog
.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for phenotrex can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/univieCUBE/phenotrex

Or download the `tarball`_:

.. code-block:: console

    $ curl -OL https://github.com/univieCUBE/phenotrex/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ make full-install


.. _Github repo: https://github.com/univieCUBE/phenotrex
.. _tarball: https://github.com/univieCUBE/phenotrex/tarball/master
