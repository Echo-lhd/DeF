# DeF

Therefore,we propose a transformer-based method for deinterlacing. The method consists of three main modules, the Feature extractor, Deinterlacing transformer and Densnet modules. 

<p align="center"><img width="100%" src="Figs/Fig1.png.png" /></p>

## Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.3](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
Clone repository

    ```bash
    git clone https://github.com/Echo-lhd/DeT.git
    ```
   

## Dataset Preparation

- Please refer to paper for more details.

## Training

- Please refer to **[configuration of training](main.py)** for more details and [pretrained models](https://drive.google.com/drive/folders/1HFZbuYq54U9mz_ngAqfW3pRMcry7XWx3?usp=sharing).  

    ```bash
    # Train on Vimeo-90K
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py
    ```

## Testing

- Please refer to **[configuration of testing](demo_Vid4.py)** for more details.

    ```bash
   # Test on Vimeo-90K
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python demo_Vid4.py
    ```
