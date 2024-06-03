# AICUP_2024_rank5


## Restormer

```
python train.py --stage 'test' --model_file './result_AICUP_ori/All.pth' --save_path 'result_AICUP_ori'
```

環境建置可參考原始 restormer 中 github 網址進行。
https://github.com/swz30/Restormer?tab=readme-ov-file

```
@inproceedings{Zamir2021Restormer,
    title={Restormer: Efficient Transformer for High-Resolution Image Restoration}, 
    author={Syed Waqas Zamir and Aditya Arora and Salman Khan and Munawar Hayat 
            and Fahad Shahbaz Khan and Ming-Hsuan Yang},
    booktitle={CVPR},
    year={2022}
}
```

## Zoomnet 
```
python main.py --model-name=ZoomNet --config=configs/zoomnet/zoomnet.py --datasets-info ./configs/_base_/dataset/dataset_configs.json --info demo
```
環境建置可參考原始 zoomnet 中 github 網址進行。
https://github.com/lartpang/ZoomNet

```
@inproceedings{ZoomNet-CVPR2022,
	title     = {Zoom In and Out: A Mixed-scale Triplet Network for Camouflaged Object Detection},
	author    = {Pang, Youwei and Zhao, Xiaoqi and Xiang, Tian-Zhu and Zhang, Lihe and Lu, Huchuan},
	booktitle = CVPR,
	year      = {2022}
}
```

## UNet and UNet_refinement
使用 python 環境 3.12.3
python train.py
