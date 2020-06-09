#!/bin/bash -x
unset PYTHONPATH
source /home/sampanna/.zshrc

## Find the directory in which this script is located.
#DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
## Find that directory's parent directory
#SOURCE="$(dirname "$DIR")"
SOURCE=/home/sampanna/workspace/bdts2/deepfigures-open
cd "$SOURCE" || exit
pwd

source activate deepfigures_3 || conda activate deepfigures_3 || exit
which python
which pip
PIP="/home/sampanna/anaconda3/envs/deepfigures_3/bin/pip"
#PYTHON="/home/sampanna/anaconda3/envs/deepfigures_3/bin/python"

rm -rf "$SOURCE"/build \
  "$SOURCE"/dist \
  "$SOURCE"/deepfigures_open.egg-info \
  "$SOURCE"/vendor/tensorboxresnet/.eggs \
  "$SOURCE"/vendor/tensorboxresnet/build \
  "$SOURCE"/vendor/tensorboxresnet/dist \
  "$SOURCE"/vendor/tensorboxresnet/tensorboxresnet.egg-info \
  "$SOURCE"/vendor/tensorboxresnet/tensorboxresnet/utils/stitch_wrapper.cpp

$PIP uninstall deepfigures-open -y
$PIP uninstall deepfigures-open -y
$PIP uninstall tensorboxresnet -y
$PIP uninstall tensorboxresnet -y
echo "Pip uninstalls complete"
python --version
cd "$SOURCE" || exit
pwd
python setup.py install
cd "$SOURCE"/vendor/tensorboxresnet && python setup.py install && cd ../..

rm -rf "$SOURCE"/build \
  "$SOURCE"/dist \
  "$SOURCE"/deepfigures_open.egg-info \
  "$SOURCE"/vendor/tensorboxresnet/.eggs \
  "$SOURCE"/vendor/tensorboxresnet/build \
  "$SOURCE"/vendor/tensorboxresnet/dist \
  "$SOURCE"/vendor/tensorboxresnet/tensorboxresnet.egg-info \
  "$SOURCE"/vendor/tensorboxresnet/tensorboxresnet/utils/stitch_wrapper.cpp
