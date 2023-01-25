pip install openmim
mim install mmcv-full==1.3.13
mim install mmdet==2.14.0
mim install mmsegmentation==0.18.0
cd mmdetection3d
pip install -e .
cd ..
pip uninstall pycocotools -y
mim install mmpycocotools
cd pypcd
python setup.py install
cd ..
pip install seaborn
# pip install vtk==8.1.2
# pip install mayavi
# apt-get update
# pip install open3d==0.11.1