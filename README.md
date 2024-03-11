# HE_transfer_learning
Implementation of Transfer Learning Using HE 

# environment
First, check if PyTorch is already installed.
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install timm
```

Before python `scp_files.py`, run this command from azure for passwordless access:
```bash
ssh-copy-id -o ProxyJump=hop20001@137.99.0.102 xix22010@192.168.10.16
```
