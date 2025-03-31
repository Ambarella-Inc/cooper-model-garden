This document is used for the LongCLIP-ViTb16 image encoder model.

## Setup

Firstly, create conda environment

```
conda create -n longclip python=3.10 -y
conda activate longclip
```

Install Pytorch with cuda support

```
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install ftfy regex tqdm
pip install -r requirements.txt
```


Clone code and install requirements

```
git clone https://github.com/beichenzbc/Long-CLIP.git
cd Long-CLIP/
git reset --hard b7acee8db07a981f38933ef01319beeb96f19a33
git apply longclip.patch
python3 export.py --pretrained checkpoints/longclip-B.pt --image_path img/framework.PNG --text "a photo of diagram" --export_path onnx_models/longclip-B16/
```

Note: 

Please refer to  **scripts/Long-CLIP/License.txt** for the license of  **longclip.patch** .