# AICUP_2024_rank5

https://drive.google.com/file/d/1OoHS2VgAaQ4hjT3Li4iJK1HVHCtaNb4I/view?usp=sharing
google drive 中已經包含四者模型的輸出以及 zoomnet, unet, unet_refinement 的權重，若要使用於 testing, 麻煩更改模型中的權重位置。

## Restormer

```
python train.py --stage 'test' --model_file './result_AICUP_ori/All.pth' --save_path 'result_AICUP_ori'
```

環境建置可參考原始 restormer 中 github 網址進行。
本次我們選用 python 版本 3.8 運行。
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
本次我們選用 python 版本 3.8 進行。
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
使用 python 環境 3.12.3 版本運行。
其中 train.py 和 train_change_weight.py 分別為原始 UNet 以及增強後的改良版 UNet.
環境所需套件已經放入至 unet/requirement.txt, 可以進行套件安裝。
```
python train.py
```
