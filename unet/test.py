from UNet import UNet
import torch
import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

weight_path = "/home/andy/AICUP/all_data_unet_epoch297.pth"
input_dir = "/home/andy/AICUP/private_test/img"
# input_dir = "/home/andy/AICUP/small_dataset/test/input"
output_dir = "/home/andy/AICUP/small_dataset/test/UNet_epoch297_private_test"



model = UNet(3,1)
model.load_state_dict(torch.load(weight_path))

os.makedirs(output_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# 读取输入目录中的所有图像文件
input_files = sorted(os.listdir(input_dir))

# 模型预测并保存结果
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for file in input_files:
    img_path = os.path.join(input_dir, file)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(img_tensor)
    prediction = prediction.squeeze(0).cpu().numpy()
    prediction = prediction.transpose(1,2,0)
    prediction = (prediction * 255).astype(np.uint8)

    output_file = os.path.splitext(file)[0] + ".png"
    output_path = os.path.join(output_dir, output_file)

    cv2.imwrite(output_path, prediction)

print("Predictions saved to:", output_dir)
