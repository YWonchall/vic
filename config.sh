pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install openmim
mim install mmcv-full==1.5.2
mim install mmdet==2.26.0
mim install mmsegmentation==0.29.1
cd mmdetection3d
pip install -v -e .
cd ../pypcd
python setup.py install
cd ..
pip install seaborn
pip install open3d==0.11.1
pip uninstall setuptools -y
pip install setuptools==52.0.0