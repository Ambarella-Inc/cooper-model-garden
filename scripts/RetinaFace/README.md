This document applies sparse training to `Retinaface`.

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
git clone https://github.com/biubug6/Pytorch_Retinaface.git
cd Pytorch_Retinaface/
git reset b984b4b775b2c4dced95c1eadd195a5c7d32a60b --hard
pip3 install -r requirements.txt
```

Then setup environment and download pretrained model and dataset as official README.md instructs.\
Finally, put train_sp_torch.py into Pytorch_Retinaface/.

## Retrain

Here we choose weights of Resnet50_Final.pth as pretrained weights and run sparse retraining on it.

```
python train.py --training_dataset ./data/widerface/train/label.txt --input_size 840 --network resnet50 --num_workers 4 \
--batch_size 8 --epoch 10 --lr 1e-5 --resume_net ./weights/Resnet50_Final.pth --save_folder ./weights/train_sp_0.8/ --sparse 0.8
```

## Export to ONNX


```
python convert_to_onnx.py -m ./weights/Resnet50_Final.pth --network resnet50 --long_side 640
```