import UNet_refinement
from UNet_refinement import UNet
import torch
import os
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

test_input_dir = "/home/andy/AICUP/private_test/input"
weight_path = "./weight/UNet_refinement_change_weight490.pth"
output_path = "./output_path"


print(output_path)


os.makedirs(output_path, exist_ok=True)
output_edge1_path = os.path.join(output_path, "edge1")
output_edge2_path = os.path.join(output_path, "edge2")
os.makedirs(output_edge1_path, exist_ok=True)
os.makedirs(output_edge2_path, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
])

def save_prediction(prediction, output_path):
    prediction = prediction.squeeze(0).cpu().numpy()
    prediction = prediction.transpose(1, 2, 0)
    prediction = (prediction * 255).astype(np.uint8)
    cv2.imwrite(output_path, prediction)


input_files = sorted(os.listdir(test_input_dir))

model = UNet(3,1)
model.load_state_dict(torch.load(weight_path))

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for file in input_files:
    img_path = os.path.join(test_input_dir, file)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        edge1, edge2 = model(img_tensor)
        mask = model(img_tensor)

    # 保存 edge1
    output_file_edge1 = os.path.splitext(file)[0] + ".png"
    output_path_edge1 = os.path.join(output_edge1_path, output_file_edge1)
    save_prediction(edge1, output_path_edge1)

    # 保存 edge2
    output_file_edge2 = os.path.splitext(file)[0] + ".png"
    output_path_edge2 = os.path.join(output_edge2_path, output_file_edge2)
    save_prediction(edge2, output_path_edge2)

    print("Predictions saved to:", output_path)