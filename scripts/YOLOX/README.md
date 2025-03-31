This document applies sparse retraining to `yolox_s`.

## Setup

Create conda environment.

```shell
 conda create -n modelgarden python=3.10 -y
 conda activate modelgarden
```

Install pytorch with cuda support.

```shell
pip3 install torch==1.13.0 torchvision==0.14.0 --index-url https://download.pytorch.org/whl/cu121
```

Set up Ambarella cvtools environment to deploy the retrained model to Ambarella CVflow chip. Please refer to Ambarella cvtools document for details.

## Retrain

Here we choose weights of **yolox-s** as pretrained weights and run sparse retraining on it.

Step1. Install YOLOX from source.

```shell
git clone git@github.com:Megvii-BaseDetection/YOLOX.git
cd YOLOX
git reset  d872c71bf63e1906ef7b7bb5a9d7a529c7a59e6a
pip3 install -v -e .  # or  python3 setup.py develop
```

Step2. Apply patch for sparse retrain

```shell
git apply yolox_sparse.patch
```

Step3. Set environment

```shell
pip3 install -r requirement.txt
```

Step4. Sparse retrain

```shell
python3 -m yolox.tools.train -n yolox-s -d 8 -b 64 --fp16 -o [--cache] --sparsity 0.5 --resume -e 1 -c ./best_ckpt.pth
```

Step5. Export to ONNX

```shell
python3 -m yolox.tools.export_onnx -c ./best_ckpt.pth -o 11 --input input --output model_output  -n yolox-s --is_sparse
```

Note:

Please refer to  scripts/YOLOX/License.txt for the license of  `yolox_sparse.patch` .

