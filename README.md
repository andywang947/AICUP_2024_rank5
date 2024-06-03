# AICUP_2024_rank5


Restormer

python train.py --stage 'test' --model_file './result_AICUP_ori/All.pth' --save_path 'result_AICUP_ori'

Zoomnet 

python main.py --model-name=ZoomNet --config=configs/zoomnet/zoomnet.py --datasets-info ./configs/_base_/dataset/dataset_configs.json --info demo

UNet and UNet_refinement

python train.py
