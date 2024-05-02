# [CVPR Vizwiz Challenge 2024]
## Environments
+ Ubuntu 18.04.6 LTS
+ CUDA Version 11.3
+ GPU: NVIDIA RTX A6000 with 48GB memory

## Prerequisites

To use the repository, we provide a conda environment.

```bash
conda update conda
conda env create -f environment.yaml
conda activate vizwiz_TTA
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

## Structure of Project

This project contains several directories. Their roles are listed as follows:
+ ./best_cfgs: the best config files for each dataset and algorithm are saved here.
+ ./datasets
  
       |-- datasets 
  	        |-- challenge
                |-- 8900
            |-- images
  
  - './images' All data downloaded from the challenge is saved. Delete the train, val, and test files. You can download it [here](https://github.com/rezaakb/VizWiz-Classification-Dataset)
  - './challenge/8900' contains 8900 pieces of test data from the challenge. If you don't have it, run the code below:
    ```bash
    pip install opencv-python==4.9.0.80
    python extract_images.py
    ```


  
## Quick start (Test)

Before starting, modify the checkpoint path in ./best_cfgs/evaluation.yaml

The pre-trained model can be found [here](https://drive.google.com/drive/folders/1UFVLyONwlqJpWE6hEw7Kqqxw2GdBo43m).

The testing results and training logs will be saved in the `./output/test-time-evaluation/"[YOUR EXPERIMENRT NAME]"`

    CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluation.py --cfg ./best_cfgs/evaluation.yaml --output_dir ./output/test-time-evaluation/"[YOUR EXPERIMENRT NAME]"`

## Train model

Before starting, modify the checkpoint path in ./best_cfgs/parallel_psedo_contrast.yaml

The testing results and training logs will be saved in the `./output/test-time-evaluation/"[YOUR EXPERIMENRT NAME]"`

    CUDA_VISIBLE_DEVICES=0,1,2,3 python challenge_test_time.py --cfg ./best_cfgs/parallel_psedo_contrast.yaml --output_dir ./test-time-evaluation/"[YOUR EXPERIMENRT NAME]"

## Acknowledgement

Our codes borrowed from [yuyongcan](https://github.com/yuyongcan/Benchmark-TTA). Thanks for their work.


