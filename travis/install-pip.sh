#!/usr/bin/env bash

set -e

echo "pip installing required python packages"
pip install -r requirements_dev.txt

python --version
