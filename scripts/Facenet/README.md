This document applies export onnx model of facenet and test result.

## Setup

Firstly, create conda environment.

```
conda create -n modelgarden python=3.10 -y
conda activate modelgarden
```

Install Pytorch with cuda support.

```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

Clone code and install requirements.

```
git clone https://github.com/bubbliiiing/facenet-pytorch
cd facenet-pytorch/
git reset --hard af7d20caeb7c164129ba8c5edae3c0e9605be169
pip3 install -r requirements.txt
```

Apply patch


```
git apply {your dir}/0001-model-garden-facenet.patch
```

## Export ONNX

Here we choose weights of facenet_mobilenet.pth as pretrained weights and export it to onnx.


```
python export_onnx.py --model-path ./model_data/facenet_mobilenet.pth --output-path ./facenet_mobilenet_org.onnx
```

Note: 

Please refer to  **scripts/Facenet/License.txt** for the license of  **0001-model-garden-facenet.patch** .