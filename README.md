# [CVPR Vizwiz Challenge 2024]
https://private-user-images.githubusercontent.com/151484020/327650333-50b0164d-c635-4b94-883b-ff188b9951dc.PNG?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTQ3MTMwMTMsIm5iZiI6MTcxNDcxMjcxMywicGF0aCI6Ii8xNTE0ODQwMjAvMzI3NjUwMzMzLTUwYjAxNjRkLWM2MzUtNGI5NC04ODNiLWZmMTg4Yjk5NTFkYy5QTkc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjQwNTAzJTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI0MDUwM1QwNTA1MTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0wZGUyNDExNTAzODQzNjgzNzRkMDA3MGQzNTM4ZWJmNWY2OTIxNTA5NTYwOTcxZTA0YjhmYzhkNmY0NzM1MWQ1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCZhY3Rvcl9pZD0wJmtleV9pZD0wJnJlcG9faWQ9MCJ9.yJmgPvv8XwVNuFhuyCf3U60WVLLv9EcpxGwfR7RRhn4
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

+ `./best_cfgs`: the best config files for each dataset and algorithm are saved here.
+ `./datasets`
  
       |-- datasets 
  	        |-- challenge
                |-- 8900
            |-- images
  
  - `./images` : All data downloaded from the challenge is saved. Delete the train, val, and test files in `./images`. You can download it [here](https://github.com/rezaakb/VizWiz-Classification-Dataset).
  - `./challenge/8900` : This file will contain 8900 pieces of test data from the challenge. If you don't have it, run the code below:
  
    ```bash
    pip install opencv-python==4.9.0.80
    python extract_images.py
    ```

## Train model

Before starting, modify the checkpoint path in `./best_cfgs/parallel_psedo_contrast.yaml`.

The testing cheakpoints and training logs will be saved in the `./output/test-time-evaluation/"[YOUR EXPERIMENRT NAME]"`.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python challenge_test_time.py --cfg ./best_cfgs/parallel_psedo_contrast.yaml --output_dir ./output/test-time-evaluation/"[YOUR EXPERIMENRT NAME]"

## Acknowledgement

Our codes borrowed from [yuyongcan](https://github.com/yuyongcan/Benchmark-TTA). Thanks for their work.


