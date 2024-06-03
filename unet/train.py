import torch
import os
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from UNet import UNet
from torch import nn
import cv2


dataset_dir = "/home/andy/AICUP/dataset"
train_input_dir = r"./train_input"
train_target_dir = r"./train_target"
test_input_dir = r"./test_input"
test_target_dir = r"./test_target"

weight_dir = r"/home/andy/AICUP/weight"
result_dir = r"/home/andy/AICUP/result"

train_name = r"normal_UNet"

weight_path = os.path.join(weight_dir,train_name)
result_path = os.path.join(result_dir,train_name)
os.makedirs(result_path, exist_ok=True)


if os.path.exists(dataset_dir):
    print("The dirctionary path is right !")

else :
    print("The dictionary doesn't exist ! ")


def check_data(data_path) :
  if os.path.exists(data_path) :
    files = os.listdir(data_path)
    print("There are ",len(files),"images in the dictionary:",data_path)
  else :
    print("The dictionary path is wrong ! You need to fix your dictionary path !")

check_data(train_input_dir)
check_data(train_target_dir)
check_data(test_input_dir)
check_data(test_target_dir)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
else :
    print("Didn't use GPU !!!")


class PairedImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_filenames = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir))]
        self.target_filenames = [os.path.join(target_dir, f) for f in sorted(os.listdir(target_dir))]
        self.transform = transform

    def __len__(self):
        return len(self.input_filenames)

    def __getitem__(self, idx):
        input_image = Image.open(self.input_filenames[idx]).convert('RGB')
        target_image = Image.open(self.target_filenames[idx]).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Dataset
train_dataset = PairedImageDataset(train_input_dir, train_target_dir, transform=transform)
test_dataset = PairedImageDataset(test_input_dir, test_target_dir, transform=transform)

# DataLoader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=True)


# Fetch a batch and display the images
data_iter = iter(train_dataloader)
images, targets = next(data_iter)

# Display images using matplotlib
fig, axes = plt.subplots(2, 4, figsize=(12, 4))

for i in range(4):
    img_input = images[i].numpy().transpose((1, 2, 0))
    img_input = img_input.clip(0, 1)

    img_target = targets[i].numpy().transpose((1, 2, 0))
    img_target = img_target.clip(0, 1)

    ax_input = axes[0, i]
    ax_input.imshow(img_input)
    ax_input.axis('off')
    ax_input.set_title(f'Input {i+1}')

    ax_target = axes[1, i]
    ax_target.imshow(img_target)
    ax_target.axis('off')
    ax_target.set_title(f'Target {i+1}')

plt.show()

print(len(train_dataloader))

# training and testing
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

def unnormalize(image):
    image = image.numpy().transpose((1, 2, 0))
    image = np.clip(image, 0, 1)
    return image

def unnormalize_gray(image):
    image = image.numpy()
    image = np.clip(image, 0, 1)
    return image

def save_prediction(prediction, output_path):
    prediction = prediction.squeeze(0).cpu().numpy()
    prediction = prediction.transpose(1, 2, 0)
    prediction = (prediction * 255).astype(np.uint8)
    cv2.imwrite(output_path, prediction)

def testing(model, test_dataloader, epoch):
    torch.save(model.state_dict(), weight_path + str(epoch) + ".pth")

    output_path = result_path + "_" + str(epoch)
    os.makedirs(output_path, exist_ok=True)
    output_edge1_path = os.path.join(output_path, "edge1")
    output_edge2_path = os.path.join(output_path, "edge2")
    os.makedirs(output_edge1_path, exist_ok=True)
    os.makedirs(output_edge2_path, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 读取输入目录中的所有图像文件
    input_files = sorted(os.listdir(test_input_dir))

    # 模型预测并保存结果
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for file in input_files:
        img_path = os.path.join(test_input_dir, file)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            # edge1, edge2 = model(img_tensor)
            mask = model(img_tensor)

        # # # 保存 edge1
        # output_file_edge1 = os.path.splitext(file)[0] + ".png"
        # output_path_edge1 = os.path.join(output_edge1_path, output_file_edge1)
        # save_prediction(edge1, output_path_edge1)

        # # # 保存 edge2
        # output_file_edge2 = os.path.splitext(file)[0] + ".png"
        # output_path_edge2 = os.path.join(output_edge2_path, output_file_edge2)
        # save_prediction(edge2, output_path_edge2)

        output_file_edge1 = os.path.splitext(file)[0] + ".png"
        output_path_edge1 = os.path.join(output_edge1_path, output_file_edge1)
        save_prediction(mask, output_path_edge1)

    print("Predictions saved to:", output_path)




def training(criterion, optimizer, model, train_dataloader, test_dataloader):
    num_epochs = 500
    testing_epoch = 10
    patience = 25  # For Early stopping use
    best_loss = float('inf')
    epochs_no_improve = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    model = model.to(device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            # edge1, edge2 = model(inputs)
            mask = model(inputs)
            loss = criterion(mask,targets)
            # loss1 = criterion(edge1, targets)
            # loss2 = criterion(edge2, targets)
            # loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_dataloader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss}')

        if epoch % testing_epoch == 0:
            testing(model, test_dataloader,epoch)
            model = model.to(device)

        if epoch > 50:
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break



model = UNet(3,1)
# model = MiniUNet(3,1)
criterion = nn.MSELoss()
# criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)
training(criterion, optimizer, model, train_dataloader, test_dataloader)
# testing(model,test_dataloader)