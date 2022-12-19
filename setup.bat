@echo off
:: if your system is windows...
:: install git for windows
:: install git lfs for windows
:: install cuda toolkit 11.6
:: install cuDNN 8.2.4
:: conda env create -f environment.yaml
:: conda activate preare

pip install -U torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
mkdir models
mkdir src
cd src
git clone https://github.com/JingyunLiang/SwinIR
git clone https://github.com/waifu-diffusion/aesthetic
