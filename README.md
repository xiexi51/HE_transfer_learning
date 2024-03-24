# HE_transfer_learning
Implementation of Transfer Learning Using HE 

## Environment Setup

1. **Check if PyTorch is already installed**: If not, install it.

    ```bash
    pip install torch torchvision torchaudio
    ```

2. **Install additional required packages**: 

    ```bash
    pip install timm
    pip install setproctitle
    ```

## Multi-node Configuration
If you intend to run the implementation across multiple nodes, you need to prepare an IP list file.

1. **Create an IP list file (`./ip_list`)**: This file should contain the IP addresses of all nodes involved in the computation, starting with the master node's IP followed by the IP addresses of slave nodes. The file format should be like:

    ```
    master_ip
    slave1_ip
    ...
    ```

## Passwordless SSH Access to Servers
If you need to access the A6000 server (192.168.10.16) without a password:

1. **Set up login by public key from the master node to the CSE gateway (137.99.0.102)**.

2. **Configure passwordless SSH to the A6000 server**: 

    ```bash
    ssh-copy-id -o ProxyJump=xix22010@137.99.0.102 xix22010@192.168.10.16
    ```
