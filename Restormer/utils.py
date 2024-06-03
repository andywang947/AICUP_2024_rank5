import argparse
import glob
import os
import cv2
import random
import copy
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional as T
import torchvision.transforms as transforms
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop,ToTensor
# from degradation_utils import Degradation
from numpy.random import RandomState
# from augmentation import random_augmentation

def parse_args():
    desc = 'Pytorch Implementation of \'Restormer: Efficient Transformer for High-Resolution Image Restoration\''
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--data_path', type=str, default='.\\data\\Training_dataset')
    # parser.add_argument('--data_name', type=str, default='Rain100L', choices=['dehaze','BSD68', 'urban100','Rain100L'])
    parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze'],
                    help='which type of degradations is training and testing for.')
    parser.add_argument('--save_path', type=str, default='result_Allweather_PrCA')
    parser.add_argument('--num_blocks', nargs='+', type=int, default=[1, 1, 1, 1], #[4, 6, 6, 8]
                        help='number of transformer blocks for each level')
    parser.add_argument('--num_heads', nargs='+', type=int, default=[1, 1, 1, 1],
                        help='number of attention heads for each level')
    parser.add_argument('--channels', nargs='+', type=int, default=[8, 16, 32, 64],
                        help='number of channels for each level')
    parser.add_argument('--expansion_factor', type=float, default=2.66, help='factor of channel expansion for GDFN')
    parser.add_argument('--num_refinement', type=int, default=4, help='number of channels for refinement stage')
    parser.add_argument('--num_iter', type=int, default=300000, help='iterations of training') #300000
    # parser.add_argument('--batch_size', nargs='+', type=int, default=[16, 16, 12, 12, 8, 8])
    parser.add_argument('--batch_size', nargs='+', type=int, default=[4, 4, 4, 4, 4, 4])
    # parser.add_argument('--batch_size', nargs='+', type=int, default=[16, 16, 4, 4, 2, 2])
    # parser.add_argument('--patch_size', nargs='+', type=int, default=[64, 64, 64, 64, 64, 64],help='patch size of each image for progressive learning')
    parser.add_argument('--patch_size', nargs='+', type=int, default= [256, 256, 256, 256, 256, 256])
    # parser.add_argument('--patch_size', nargs='+', type=int, default=[64, 64, 128, 128, 256, 256])

    parser.add_argument('--lr', type=float, default=0.000003, help='initial learning rate') #0.0003
    parser.add_argument('--milestone', nargs='+', type=int, default=[92000, 156000, 204000, 240000, 276000],
                        help='when to change patch size and batch size')
    parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (-1 for no manual seed)')
    parser.add_argument('--stage', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--val_iter', type=int, default=2500)
    parser.add_argument('--model_file', type=str, default=None, help='path of pre-trained model file')

    return init_args(parser.parse_args())


class Config(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.save_path = args.save_path
        self.num_blocks = args.num_blocks
        self.num_heads = args.num_heads
        self.channels = args.channels
        self.expansion_factor = args.expansion_factor
        self.num_refinement = args.num_refinement
        self.num_iter = args.num_iter
        self.batch_size = args.batch_size
        self.patch_size = args.patch_size
        self.lr = args.lr
        self.milestone = args.milestone
        self.workers = args.workers
        self.model_file = args.model_file
        self.stage = args.stage
        self.val_iter = args.val_iter
        self.de_type = args.de_type


def init_args(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.seed >= 0:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


    return Config(args)


class GrayLoss(nn.Module):
    def __init__(self):
        super(GrayLoss,self).__init__()
        self.l1 = nn.L1Loss().cuda()

    def forward(self, x):
        x = torch.sigmoid(x)
        y = torch.ones_like(x) / 2.
        return 1 / self.l1(x, y)

class OHCeLoss(nn.Module):
    def __init__(self):
        super(OHCeLoss,self).__init__()
    def forward(self,pred,onehot_label):
        pred = pred.squeeze()
        onehot_label = onehot_label.squeeze()
        N = pred.size(0)
        # log_prob = F.log_softmax(pred, dim=1)
        log_prob = torch.log(pred)
        loss = -torch.sum(log_prob * onehot_label) / N
        return loss

class Cosine_similarity(nn.Module):
    def __init__(self,dataset):
        super(Cosine_similarity,self).__init__()
        self.dataset = dataset
    def forward(self,x,p):
        if self.dataset == 'allweather':
            x = x / x.norm(dim=-1, keepdim=True)
            #clean
            p_clean = p[0,:].unsqueeze(0)
            p_clean = p_clean / p_clean.norm(dim=-1, keepdim=True)
            similarity = ( x @ (p_clean.T))
            sim_clean = torch.sigmoid(similarity)
            #raindrop
            p_raindrop = p[1,:].unsqueeze(0)
            p_raindrop = p_raindrop / p_raindrop.norm(dim=-1, keepdim=True)
            similarity = ( x @ (p_raindrop.T))
            sim_raindrop = torch.sigmoid(similarity)

            #rainhaze
            p_rainhaze = p[2,:].unsqueeze(0)
            p_rainhaze = p_rainhaze / p_rainhaze.norm(dim=-1, keepdim=True)
            similarity = ( x @ (p_rainhaze.T))
            sim_rainhaze = torch.sigmoid(similarity)

            #snow
            p_snow = p[3,:].unsqueeze(0)
            p_snow = p_snow / p_snow.norm(dim=-1, keepdim=True)
            similarity = ( x @ (p_snow.T))
            sim_snow = torch.sigmoid(similarity)

            loss = -torch.log(sim_clean/(sim_clean + sim_raindrop +sim_rainhaze +sim_snow))
        elif self.dataset == 'weatherstream':
            x = x / x.norm(dim=-1, keepdim=True)
            #clean
            p_clean = p[0,:].unsqueeze(0)
            p_clean = p_clean / p_clean.norm(dim=-1, keepdim=True)
            similarity = ( x @ (p_clean.T))
            sim_clean = torch.sigmoid(similarity)
            # #raindrop
            # p_raindrop = p[1,:].unsqueeze(0)
            # p_raindrop = p_raindrop / p_raindrop.norm(dim=-1, keepdim=True)
            # similarity = ( x @ (p_raindrop.T))
            # sim_raindrop = torch.sigmoid(similarity)

            #rainhaze
            p_rainhaze = p[2,:].unsqueeze(0)
            p_rainhaze = p_rainhaze / p_rainhaze.norm(dim=-1, keepdim=True)
            similarity = ( x @ (p_rainhaze.T))
            sim_rainhaze = torch.sigmoid(similarity)

            # #snow
            # p_snow = p[3,:].unsqueeze(0)
            # p_snow = p_snow / p_snow.norm(dim=-1, keepdim=True)
            # similarity = ( x @ (p_snow.T))
            # sim_snow = torch.sigmoid(similarity)

            loss = -torch.log(sim_clean/(sim_clean + sim_rainhaze))

        return loss

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_num_in_millions = total_num / 1000000
    trainable_num_in_millions = trainable_num / 1000000
    print('{:<30}  {:.2f}M'.format('Number of parameters: ', total_num_in_millions))
    print('{:<30}  {:.2f}M'.format('Number of Trainable parameters: ', trainable_num_in_millions))
    # return {'Total': total_num, 'Trainable': trainable_num}

def pad_image_needed(img, size):
    # width, height = T.get_image_size(img)
    width, height =  img.shape[1:]
    img = img.unsqueeze(0)
    if width < size[1]:
        pad_num = size[1] - width
        if pad_num == width:
            target_size = (width+1, height)
            img = F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
        # if pad_num % 2 !=0:
        #     pad_num = pad_num+1 dsa
        img = F.pad(img, (0,0,0,pad_num), 'reflect')

    if height < size[0]:
        pad_num = size[0] - height
        if pad_num == height:
            target_size = (width, height+1)
            img = F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)
        # if pad_num % 2 !=0:
        #     pad_num = pad_num+1 
        img = F.pad(img,(0,pad_num,0, 0), 'reflect')
    return img.squeeze(0)

def extend_data(data_list, extend_num):
    extend_list = []
    tempH = []
    tempS = []
    for img in data_list:
        if ("_rain" in img):
            extend_list.append(img)
        elif ("im_" in img):
            tempH.append(img)
        else:
            tempS.append(img)
    print(f'RainDrop :{len(extend_list)}')
    print(f'Haze :{len(tempH)}')
    print(f'Snow :{len(tempS)}')
    extend_lists = []
    for _ in range(extend_num):
        extend_lists += copy.deepcopy(extend_list)

    return data_list + extend_lists



# class PromptTrainDataset(Dataset):
#     def __init__(self, args, data_type, patch_size=None, length=None):
#         super(PromptTrainDataset, self).__init__()
#         self.args = args
#         self.rs_ids = []
#         self.hazy_ids = []
#         self.D = Degradation(args)
#         self.de_temp = 0
#         self.de_type = args.de_type
#         print(self.de_type)
#         self.toTensor = ToTensor()
#         self.rand_state = RandomState(66)
#         self.de_dict = {'denoise_15': 0, 'denoise_25': 1, 'denoise_50': 2, 'derain': 3, 'dehaze': 4, 'deblur' : 5}
#         self.data_type, self.patch_size = data_type, patch_size

#         self._init_ids()
#         self._merge_ids()

#         # make sure the length of training and testing different
#         self.num = len(self.sample_ids)
#         self.sample_num = length if data_type == 'train' else self.num


#     def _init_ids(self):
#         if 'denoise_15' in self.de_type or 'denoise_25' in self.de_type or 'denoise_50' in self.de_type:
#             self._init_clean_ids()
#         if 'derain' in self.de_type:
#             self._init_rs_ids()
#         if 'dehaze' in self.de_type:
#             self._init_hazy_ids()

#         random.shuffle(self.de_type)

#     def _init_clean_ids(self):
#         data_path = self.args.data_path + '/Train/Denoise/target'
#         clean_ids = glob.glob(str(data_path) + "/*")
#         # temp_ids = []
#         # temp_ids+= [id_.strip() for id_ in open(ref_file)]
#         # clean_ids = []
#         # name_list = os.listdir(self.args.denoise_dir)
#         # clean_ids += [self.args.denoise_dir + id_ for id_ in name_list if id_.strip() in temp_ids]

#         if 'denoise_15' in self.de_type:
#             self.s15_ids = [{"clean_id": x,"de_type":0} for x in clean_ids]
#             self.s15_ids = self.s15_ids * 3
#             random.shuffle(self.s15_ids)
#             self.s15_counter = 0
#         if 'denoise_25' in self.de_type:
#             self.s25_ids = [{"clean_id": x,"de_type":1} for x in clean_ids]
#             self.s25_ids = self.s25_ids * 3
#             random.shuffle(self.s25_ids)
#             self.s25_counter = 0
#         if 'denoise_50' in self.de_type:
#             self.s50_ids = [{"clean_id": x,"de_type":2} for x in clean_ids]
#             self.s50_ids = self.s50_ids * 3
#             random.shuffle(self.s50_ids)
#             self.s50_counter = 0

#         self.num_clean = len(clean_ids)
#         print("Total Denoise Ids : {}".format(self.num_clean))

#     def _init_hazy_ids(self):
#         hazy_path = self.args.data_path + "/Train/Dehaze/input"
#         temp_ids = glob.glob(str(hazy_path) + "/*")

#         # temp_ids = []
#         # hazy = self.args.data_file_dir + "hazy/hazy_outside.txt"
#         # temp_ids+= [self.args.dehaze_dir + id_.strip() for id_ in open(hazy)]
#         self.hazy_ids = [{"clean_id" : x,"de_type":4} for x in temp_ids]

#         self.hazy_counter = 0

#         self.num_hazy = len(self.hazy_ids)
#         print("Total Hazy Ids : {}".format(self.num_hazy))

#     def _init_rs_ids(self):
#         rain_path = self.args.data_path + "/Train/Derain/input"
#         temp_ids = glob.glob(str(rain_path) + "/*")
#         # temp_ids = []
#         # rs = self.args.data_file_dir + "rainy/rainTrain.txt"
#         # temp_ids+= [self.args.derain_dir + id_.strip() for id_ in open(rs)]
#         self.rs_ids = [{"clean_id":x,"de_type":3} for x in temp_ids]
#         self.rs_ids = self.rs_ids * 120

#         self.rl_counter = 0
#         self.num_rl = len(self.rs_ids)
#         print("Total Rainy Ids : {}".format(self.num_rl))


#     def _get_gt_name(self, rainy_name):
#         gt_name = rainy_name.replace('input','target')

#         # gt_name = rainy_name.split("rainy")[0] + 'gt/norain-' + rainy_name.split('rain-')[-1]
#         return gt_name

#     def _get_nonhazy_name(self, hazy_name):
#         dir_name = hazy_name.split("input")[0] + 'target/'
#         name = hazy_name.split('/')[-1].split('_')[0]
#         suffix = '.' + hazy_name.split('.')[-1]
#         nonhazy_name = dir_name + name + suffix
#         return nonhazy_name

#     def _merge_ids(self):
#         self.sample_ids = []
#         if "denoise_15" in self.de_type:
#             self.sample_ids += self.s15_ids
#             self.sample_ids += self.s25_ids
#             self.sample_ids += self.s50_ids
#         if "derain" in self.de_type:
#             self.sample_ids+= self.rs_ids

#         if "dehaze" in self.de_type:
#             self.sample_ids+= self.hazy_ids
#         print(f'Num of Train Data : {len(self.sample_ids)}')

#     def crop(self, img,patch_size):
#         h, w, c = img.shape
#         p_h, p_w = patch_size,patch_size

#         r = self.rand_state.randint(0, h - p_h)
#         c = self.rand_state.randint(0, w - p_w)
#         O = img[r: r + p_h, c : c + p_w]
#         return O,r,c

#     def __getitem__(self, idx):
#         sample = self.sample_ids[idx % self.num]
#         de_id = sample["de_type"]
#         # rain_feature = get_clip_img_embedding(sample["clean_id"])
#         if de_id < 3:
#             clean_id = sample["clean_id"]
#             clean_name = clean_id.split("/")[-1]
#             clean_img = np.array(Image.open(clean_id).convert('RGB'))

#             if de_id == 0:
#                 # clean_id = sample["clean_id"]
#                 degrad_id = clean_id.replace('target','input_15')
#             elif de_id == 1:
#                 # clean_id = sample["clean_id"]
#                 degrad_id = clean_id.replace('target','input_25')
#             elif de_id == 2:
#                 # clean_id = sample["clean_id"]
#                 degrad_id = clean_id.replace('target','input_50')

#             degrad_img = np.array(Image.open(degrad_id).convert('RGB'))

#             # clean_patch,row,col= self.crop(clean_img,self.patch_size)
#             # clean_patch = random_augmentation(clean_patch)[0]
#             # degrad_patch = self.D.single_degrade(clean_patch, de_id)

#         else:
#             if de_id == 3:
#                 # Rain Streak Removal
#                 degrad_id = sample["clean_id"]
#                 degrad_img = np.array(Image.open(sample["clean_id"]).convert('RGB'))
#                 # degrad_img = crop_img(np.array(Image.open(sample["clean_id"]).convert('RGB')), base=16)
#                 clean_name = self._get_gt_name(sample["clean_id"])
#                 clean_img = np.array(Image.open(clean_name).convert('RGB'))
#             elif de_id == 4:
#                 # Dehazing with SOTS outdoor training set
#                 degrad_id = sample["clean_id"]
#                 degrad_img = np.array(Image.open(sample["clean_id"]).convert('RGB'))
#                 clean_name = self._get_nonhazy_name(sample["clean_id"])
#                 clean_img = np.array(Image.open(clean_name).convert('RGB'))

#         clean_patch,row,col= self.crop(clean_img,self.patch_size)
#         degrad_patch = degrad_img[row: row + self.patch_size, col : col + self.patch_size]
#         clean_patch,degrad_patch = random_augmentation(clean_patch,degrad_patch)

#         clean_patch = self.toTensor(clean_patch)
#         degrad_patch = self.toTensor(degrad_patch)

#         clip_img = self.get_clip_img(degrad_id)
#         return degrad_patch, clean_patch, clean_name, clip_img

#     def __len__(self):
#         return self.sample_num

#     def get_clip_img(self,image_path):
#         img = cv2.imread(image_path)
#         img = cv2.resize(img, (512,512))
#         flip = random.choice([0, 1])
#         if flip==1:
#             img = cv2.flip(img, 1)
#         flip = random.choice([0, 1])
#         if flip==1:
#             img = cv2.flip(img, 0)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.astype(np.float32) / 255.0
#         img = torch.from_numpy(img).float()
#         img_resize = transforms.Resize((224,224))
#         clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#         img=img.permute(2,0,1)
#         img=img_resize(img)
#         img=clip_normalizer((img.reshape(1,3,224,224)))
#         return img




# class RainHazeDataset(Dataset):
#     def __init__(self,args,testData_path,mode=None):
#         super().__init__()
#         self.toTensor = ToTensor()
#         self.mode = mode.lower()
#         self.test_ids = glob.glob(str(testData_path) + "/*")


#         print(f'Num of RainHaze Data : {len(self.test_ids)}')

#     def _get_nonhazy_name(self, hazy_name):
#         dir_name = hazy_name.split("input")[0] + 'target/'
#         name = hazy_name.split('/')[-1].split('_')[0]
#         # suffix = '.' + hazy_name.split('.')[-1]
#         nonhazy_name = dir_name + name + '.png'
#         return nonhazy_name

#     def __len__(self):
#         return len(self.test_ids)

#     def __getitem__(self, idx):
#         if self.mode == 'derain':
#             image_name = os.path.basename(self.test_ids[idx])
#             rain = self.toTensor(Image.open(self.test_ids[idx]))
#             norain_path = self.test_ids[idx].replace('input','target')
#             norain = self.toTensor(Image.open(norain_path))
#             image_name = os.path.basename(norain_path)
#             # rain_feature = get_clip_img_embedding(self.test_ids[idx])
#         elif self.mode == 'dehaze':
#             rain = self.toTensor(Image.open(self.test_ids[idx]))
#             clean_id = self._get_nonhazy_name(self.test_ids[idx])
#             norain = self.toTensor(Image.open(clean_id))
#             image_name = os.path.basename(clean_id)
#             # rain_feature = get_clip_img_embedding(self.test_ids[idx])
#         degrad_id = self.test_ids[idx]
#         # h, w = rain.shape[1:]

#         # # padding in case images are not multiples of 8
#         # new_h, new_w = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
#         # pad_h = new_h - h if h % 8 != 0 else 0
#         # pad_w = new_w - w if w % 8 != 0 else 0
#         # rain = F.pad(rain, (0, pad_w, 0, pad_h), 'reflect')
#         # norain = F.pad(norain, (0, pad_w, 0, pad_h), 'reflect')
#         clip_img = self.get_clip_img(degrad_id)
#         return rain, norain, image_name,clip_img

#     def get_clip_img(self,image_path):
#         img = cv2.imread(image_path)
#         img = cv2.resize(img, (512,512))
#         flip = random.choice([0, 1])
#         if flip==1:
#             img = cv2.flip(img, 1)
#         flip = random.choice([0, 1])
#         if flip==1:
#             img = cv2.flip(img, 0)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.astype(np.float32) / 255.0
#         img = torch.from_numpy(img).float()
#         img_resize = transforms.Resize((224,224))
#         clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#         img=img.permute(2,0,1)
#         img=img_resize(img)
#         img=clip_normalizer((img.reshape(1,3,224,224)))
#         return img




# class DenoiseTestDataset(Dataset):
#     def __init__(self, args,testData_path,sigma = 15):
#         super(DenoiseTestDataset, self).__init__()
#         self.args = args
#         self.clean_ids = []
#         self.sigma = sigma
#         self.toTensor = ToTensor()

#         self.clean_ids = glob.glob(str(testData_path) + "/*")

#         self.num_clean = len(self.clean_ids)

#         print(f'Num of Denoise Data : {self.num_clean}')

#     # def _add_gaussian_noise(self, clean_patch):
#     #     noise = np.random.randn(*clean_patch.shape)
#     #     noisy_patch = np.clip(clean_patch + noise * self.sigma, 0, 255).astype(np.uint8)
#     #     return noisy_patch, clean_patch

#     # def set_sigma(self, sigma):
#     #     self.sigma = sigma

#     def __getitem__(self, clean_id):
#         clean_img = self.toTensor(Image.open(self.clean_ids[clean_id]).convert('RGB'))
#         clean_name = os.path.basename(self.clean_ids[clean_id])
#         if self.sigma == 15 :
#             degrad_id = self.clean_ids[clean_id].replace('BSD68','BSD68_15')
#             noisy_img = self.toTensor(Image.open(degrad_id).convert('RGB'))
#         elif self.sigma == 25 :
#             degrad_id = self.clean_ids[clean_id].replace('BSD68','BSD68_25')
#             noisy_img = self.toTensor(Image.open(degrad_id).convert('RGB'))
#         elif self.sigma == 50 :
#             degrad_id = self.clean_ids[clean_id].replace('BSD68','BSD68_50')
#             noisy_img = self.toTensor(Image.open(degrad_id).convert('RGB'))


#         # noisy_img_np, _ = self._add_gaussian_noise(clean_img)
#         # rain_feature = get_clip_img_embedding(noisy_img_np , 'denoise')
#         clip_img = self.get_clip_img(degrad_id)

#         return noisy_img, clean_img ,clean_name,clip_img

#     def __len__(self):
#         return self.num_clean

#     def get_clip_img(self,image_path):
#         img = cv2.imread(image_path)
#         img = cv2.resize(img, (512,512))
#         flip = random.choice([0, 1])
#         if flip==1:
#             img = cv2.flip(img, 1)
#         flip = random.choice([0, 1])
#         if flip==1:
#             img = cv2.flip(img, 0)
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = img.astype(np.float32) / 255.0
#         img = torch.from_numpy(img).float()
#         img_resize = transforms.Resize((224,224))
#         clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
#         img=img.permute(2,0,1)
#         img=img_resize(img)
#         img=clip_normalizer((img.reshape(1,3,224,224)))
#         return img


class RainDataset(Dataset):
    def __init__(self, args,data_type=None, patch_size=None, length=None , test_path = None):
        super().__init__()
        self.args = args
        self.data_type, self.patch_size = data_type, patch_size
        self.toTensor = ToTensor()
        self.rand_state = RandomState(66)


        if data_type == 'train':
            data_path = self.args.data_path + '/input'
            self.rain_images = glob.glob(str(data_path) + "/*")
            # self.rain_images = extend_data(self.rain_images, 9)
        else:
            data_path = str(test_path) + '/input'
            self.rain_images = glob.glob(str(data_path) + "/*")
        # make sure the length of training and testing different
        self.num = len(self.rain_images)
        self.sample_num = length if data_type == 'train' else self.num
        print(f'Num of Data : {self.sample_num}')



    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        image_name = os.path.basename(self.rain_images[idx % self.num])
        rain = T.to_tensor(Image.open(self.rain_images[idx % self.num]).convert('RGB'))
        norain =  T.to_tensor(Image.open(self.rain_images[idx % self.num].replace('input','target').replace('jpg','png')).convert('RGB'))

        if self.data_type == 'train':
            # make sure the image could be cropped
            # rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
            # norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
            # i, j, th, tw = RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
            # rain = T.crop(rain, i, j, th, tw)
            # norain = T.crop(norain, i, j, th, tw)

            rain = rain.unsqueeze(0)
            norain = norain.unsqueeze(0)

            factor = 8
            h,w = rain.shape[2], rain.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            rain = F.pad(rain, (0,padw,0,padh), 'reflect')
            # print(rain.shape)
            norain = F.pad(norain, (0, padw, 0, padh), 'reflect')
            # print(norain.shape)
            # print()
            rain = rain.squeeze(0)
            norain = norain.squeeze(0)

            if torch.rand(1) < 0.5:
                rain = T.hflip(rain)
                norain = T.hflip(norain)
            if torch.rand(1) < 0.5:
                rain = T.vflip(rain)
                norain = T.vflip(norain)
        # else:
        #     # padding in case images are not multiples of 8
        #     new_h, new_w = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
        #     pad_h = new_h - h if h % 8 != 0 else 0
        #     pad_w = new_w - w if w % 8 != 0 else 0
        #     rain = F.pad(rain, (0, pad_w, 0, pad_h), 'reflect')
        #     norain = F.pad(norain, (0, pad_w, 0, pad_h), 'reflect')

        # clean_patch,row,col= self.crop(clean_img,self.patch_size)
        # degrad_patch = degrad_img[row: row + self.patch_size, col : col + self.patch_size]
        # clean_patch,degrad_patch = random_augmentation(clean_patch,degrad_patch)

        # clean_patch = self.toTensor(clean_patch)
        # degrad_patch = self.toTensor(degrad_patch)

        # clip_img = self.get_clip_img(rain)

		# label
        # if ("target" in self.rain_images[idx % self.num]):
        #     label=torch.from_numpy(np.array([1,0,0,0]))
        # elif ("_rain" in self.rain_images[idx % self.num]):
		# 	# print("raindrop!!")
        #     label=torch.from_numpy(np.array([0,1,0,0]))
        # elif ("im_" in self.rain_images[idx % self.num]):
        #     label=torch.from_numpy(np.array([0,0,1,0]))
        # else:
        #     label=torch.from_numpy(np.array([0,0,0,1]))

        return rain, norain, image_name # ,clip_img ,label


    # def get_clip_img(self,img):
    #     # img = Image.open(image_path).convert('RGB')
    #     # img = T.to_tensor(img)
    #     img_resize = transforms.Resize((224,224))
    #     clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    #     img=img_resize(img)
    #     img=clip_normalizer((img.reshape(3,224,224)))
    #     return img


class RealRainDataset(Dataset):
    def __init__(self, args,data_type=None, patch_size=None, length=None , test_path = None):
        super().__init__()
        self.args = args
        self.data_type, self.patch_size = data_type, patch_size
        self.toTensor = ToTensor()
        self.rand_state = RandomState(66)
        self.rain_images = []




        if data_type == 'train':
            data_path = self.args.data_path + '\\train\\input'
            # self.rain_images = glob.glob(str(data_path) + "/*")
            # 获取 input 目录下的直接子目录
            subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
            # 遍历每个子目录
            for subdir in subdirs:
                subdir_path = os.path.join(data_path, subdir)
                # 获取子目录中的所有文件
                files = glob.glob(os.path.join(subdir_path, "*"))
                # 添加文件到 rain_images 列表
                self.rain_images.extend(files)


        else:
            data_path = str(test_path) + '\\input'
            # self.rain_images = glob.glob(str(data_path) + "/*")
            # 获取 input 目录下的直接子目录
            subdirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
            # 遍历每个子目录
            for subdir in subdirs:
                subdir_path = os.path.join(data_path, subdir)
                # 获取子目录中的所有文件
                files = glob.glob(os.path.join(subdir_path, "*"))
                # 添加文件到 rain_images 列表
                self.rain_images.extend(files)

        # make sure the length of training and testing different
        self.num = len(self.rain_images)
        self.sample_num = length if data_type == 'train' else self.num
        print(f'Num of Data : {self.sample_num}')



    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        image_name = os.path.basename(self.rain_images[idx % self.num])
        rain = T.to_tensor(Image.open(self.rain_images[idx % self.num]).convert('RGB'))
        file_name = os.path.basename(self.rain_images[idx % self.num])
        norain =  T.to_tensor(Image.open(self.rain_images[idx % self.num].replace('input','target').replace(file_name,'gt.png')).convert('RGB'))

        h, w = rain.shape[1:]
        if self.data_type == 'train':
            # make sure the image could be cropped
            rain = pad_image_needed(rain, (self.patch_size, self.patch_size))
            norain = pad_image_needed(norain, (self.patch_size, self.patch_size))
            i, j, th, tw = RandomCrop.get_params(rain, (self.patch_size, self.patch_size))
            rain = T.crop(rain, i, j, th, tw)
            norain = T.crop(norain, i, j, th, tw)
            if torch.rand(1) < 0.5:
                rain = T.hflip(rain)
                norain = T.hflip(norain)
            if torch.rand(1) < 0.5:
                rain = T.vflip(rain)
                norain = T.vflip(norain)
        # else:
        #     # padding in case images are not multiples of 8
        #     new_h, new_w = ((h + 8) // 8) * 8, ((w + 8) // 8) * 8
        #     pad_h = new_h - h if h % 8 != 0 else 0
        #     pad_w = new_w - w if w % 8 != 0 else 0
        #     rain = F.pad(rain, (0, pad_w, 0, pad_h), 'reflect')
        #     norain = F.pad(norain, (0, pad_w, 0, pad_h), 'reflect')

        # clean_patch,row,col= self.crop(clean_img,self.patch_size)
        # degrad_patch = degrad_img[row: row + self.patch_size, col : col + self.patch_size]
        # clean_patch,degrad_patch = random_augmentation(clean_patch,degrad_patch)

        # clean_patch = self.toTensor(clean_patch)
        # degrad_patch = self.toTensor(degrad_patch)

        clip_img = self.get_clip_img(rain)

		# label
        if ("target" in self.rain_images[idx % self.num]):
            label=torch.from_numpy(np.array([1,0,0,0]))
        elif ("_rain" in self.rain_images[idx % self.num]):
			# print("raindrop!!")
            label=torch.from_numpy(np.array([0,1,0,0]))
        elif ("im_" in self.rain_images[idx % self.num]):
            label=torch.from_numpy(np.array([0,0,1,0]))
        else:
            label=torch.from_numpy(np.array([0,0,0,1]))

        return rain, norain, image_name ,clip_img ,label


    def get_clip_img(self,img):
        # img = Image.open(image_path).convert('RGB')
        # img = T.to_tensor(img)
        img_resize = transforms.Resize((224,224))
        clip_normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        img=img_resize(img)
        img=clip_normalizer((img.reshape(3,224,224)))
        return img


def rgb_to_y(x):
    rgb_to_grey = torch.tensor([0.256789, 0.504129, 0.097906], dtype=x.dtype, device=x.device).view(1, -1, 1, 1)
    return torch.sum(x * rgb_to_grey, dim=1, keepdim=True).add(16.0)


def psnr(x, y, data_range=255.0):
    # x, y = x / data_range, y / data_range
    # mse = torch.mean((x - y) ** 2)
    # score = - 10 * torch.log10(mse)
    # return score

    x = x.cpu().numpy()
    y = y.cpu().numpy()
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    mse = np.mean((x - y)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(x, y, kernel_size=11, kernel_sigma=1.5, data_range=255.0, k1=0.01, k2=0.03):
    x, y = x / data_range, y / data_range
    # average pool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if f > 1:
        x, y = F.avg_pool2d(x, kernel_size=f), F.avg_pool2d(y, kernel_size=f)

    # gaussian filter
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device)
    coords -= (kernel_size - 1) / 2.0
    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * kernel_sigma ** 2)).exp()
    g /= g.sum()
    kernel = g.unsqueeze(0).repeat(x.size(1), 1, 1, 1)

    # compute
    c1, c2 = k1 ** 2, k2 ** 2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx, mu_yy, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_xx = F.conv2d(x ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y ** 2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)
    # structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs
    return ss.mean()
