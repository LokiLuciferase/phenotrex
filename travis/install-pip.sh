#!/usr/bin/env bash

set -e

# Alternative to "conda init bash"
source "$HOME/miniconda/etc/profile.d/conda.sh"

conda activate test
hash -r

# Work-around for a bug in PyTorch/MKLDNN on Linux and AVX512 systems:
# Fetch the pytorch-nightly, until the official release.
if [[ "$TRAVIS_OS_NAME" == 'linux' ]]; then
  echo "install and upgrade PyTorch nightly"
  conda install --yes pytorch cpuonly -c pytorch-nightly
  echo "pip installing required python packages"
  pip install -r requirements/dev.txt
  python -c "import torch; print(f'PyTorch version = {torch.__version__}')"
elif [[ "$TRAVIS_OS_NAME" == 'osx' ]]; then
  echo "pip installing required python packages"
  pip install -r requirements/dev.txt
fi

python --version
