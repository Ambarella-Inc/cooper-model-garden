This document applies sparse retraining to Resnet50 with [Torchvision](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html).

## Setup

Firstly create conda environment.

```
conda create -n modelgarden python=3.10 -y
conda activate modelgarden
```

Install Pytorch with cuda support.

```
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121
```

Install requirements.

```
pip install -r requirements.txt
```

## Retrain

Here we choose ResNet50_Weights.IMAGENET1K_V1 as pretrained weights and run sparse retraining on it.

Train script.

```
    torchrun --nproc_per_node=2  /path/to/torchvision/classification/train.py \
    	--data-path <path/to/imagenet1k>	\
    	--model resnet50	\
    	--epochs 20	\
    	--lr 5e-4	\
    	--weights ResNet50_Weights.IMAGENET1K_V1	\
    	--output-dir < path/to/output/folder > \
    	--batch-size 256	\
    	--lr-scheduler cosineannealinglr	\
    	--print-freq 100	\
    	--sparsity 0.6
```

Export to ONNX model.

```
    python3 /path/to/export.py	\
	--model resnet50	\
	--weights ResNet50_Weights.IMAGENET1K_V1	\
	--resume </path/to/retrained/checkpoint/>	\
	--output </path/to/export/onnx/model>\
	--is-sparse
```