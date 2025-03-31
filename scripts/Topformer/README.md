# Guide line for setup environment and export model to ONNX format


The author of this Topformer use an old version third party python library called "mmcv", which need some dependence for pytorch, mmsegmentation, cuda, etc. It is suggested to use a docker to setup the development environment for this project. For more information, please refer to: https://github.com/hustvl/Topformer.

#1 (Optional) get docker image from nividia docker hub.

```bash
sudo docker pull nvcr.io/nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04
sudo docker run -it --gpus all --name=cuda10.1_dev nvcr.io/nvidia/cuda:10.1-cudnn8-devel-ubuntu18.04 /bin/bash
```

#2 Install dependency

```bash
apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-dev curl

apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa -y
apt update
apt install python3.8 -y

apt-get install python3-pip
```

#3 Install python third party.

```bash
python3 -m pip install mmcv-full==1.3.13 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6/index.html

python3 -m pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/open-mmlab/mmsegmentation.git <user_path>/mmsegmentation
cd <user_path>/mmsegmentation
git reset --hard 464ebab53e5febb56ac5ffe7f190e8f87bdae7de
python3 -m pip install --no-cache-dir -e .
```

For more information about the old version mmcv installation, visit https://mmcv.readthedocs.io/zh-cn/1.x/get_started/installation.html#pip.

#4 Get source code and model from github

```bash
cd <user_path>
git clone https://github.com/hustvl/Topformer
```

The pytorch model link is also in the github page: https://github.com/hustvl/Topformer

#5 Export pytorch model to ONNX

```bash
cd <user_path>/Topformer (which is the cloned source code path)
python3 tools/convert2onnx.py local_configs/topformer/topformer_base_512x512_160k_4x8_ade20k.py --checkpoint weight/TopFormer-B_512x512_4x8_160k-39.2.pth --output-file TopFormer-B_512x512_4x8_160k.onnx
```

