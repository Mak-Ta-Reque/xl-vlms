#!/bin/bash
export PATH=/root/anaconda3/bin:\$PATH
apt update ;  apt clean
source activate xl_vlm
python -m pip install --upgrade pip
pip install -r requirements.txt
conda deactivate
