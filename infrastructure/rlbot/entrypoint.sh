#!/bin/bash

TF_V="2.13.0"

if test -d "./.venv"; then
    echo "Conda environments installed"
else
    echo "install linux packages"
    conda create --prefix ${PY_PATH} python=${PY_V} -y
    eval "$(conda shell.bash hook)"
    conda activate ${PY_PATH}

    # Install linux tensorflow
    echo "installing tensorflow"
    pip install --upgrade pip
    conda install -c conda-forge cudatoolkit=11.8.0 git -y
    python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==${TF_V}
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    # Verify install:
    # python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    echo "tensorflow installed"

    # Install linux python packages
    echo "installing other packages"
    conda install -c anaconda -y
    poetry init --python=~${PY_V}
    poetry add --lock tensorflow=${TF_V}
    poetry install --with rl
    pre-commit install

fi

# build and install the cli
source activate ./.venv
poetry build
pip install --user ./dist/rlbot-0.0.1-py3-none-any.whl --force-reinstall --no-deps
pip uninstall rlbot -y
poetry install

# See documentation for why it needs to be a fresh install each time
# Installing wine python
echo "installing wine python"
if ! test -f "python-${PY_V}-amd64.exe"; then
wget https://www.python.org/ftp/python/${PY_V}/python-${PY_V}-amd64.exe
fi
wine python-${PY_V}-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
# && rm python-${PY_V}-amd64.exe \
echo "Python installed successfully"

# Installing wine python packages
echo "installing python packages" \
&& wine python -m pip install poetry \
&& wine poetry config virtualenvs.create false \
&& wine poetry config virtualenvs.in-project false \
&& wine poetry install --only mt5

# Installing MT5
echo "Installing MT5"
if ! test -f "mt5setup.exe"; then
wget https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe
fi
if ! test -f "/workspaces/rlbot/data/platforms/mt5/0/MetaTrader 5/terminal64.exe"; then
    wine mt5setup.exe /auto
    mkdir -p /workspaces/rlbot/data/platforms/mt5/0
    cp -r "/root/.wine/drive_c/Program Files/MetaTrader 5" /workspaces/rlbot/data/platforms/mt5/0
fi
# && rm mt5setup.exe \
echo "MT5 installed successfully"
