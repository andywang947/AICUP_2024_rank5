import os
import pdb
import random
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torchvision.transforms as transforms
# import pandas as pd
import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.model import Restormer
# from models.segnet import segnet
from utils import parse_args, rgb_to_y, psnr, ssim ,RainDataset , get_parameter_number, GrayLoss
# import clip

from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
# from CLIP_LIT.prompt import L_clip_MSE,Prompts_L

########################################################

factor = 8
def test_loop(net, data_loader, num_iter ,data_name):
    net.eval()

    total_psnr, total_ssim, count = 0.0, 0.0, 0
    psnr_rgbs = 0
    ssim_rgbs = 0
    with torch.no_grad():
        test_bar = tqdm(data_loader, initial=1, dynamic_ncols=True)

        for rain, norain, name in test_bar:

            rain, norain = rain.cuda(), norain.cuda()

            # Padding in case images are not multiples of 8
            if len(rain.shape)==3:
                rain = rain.unsqueeze(0)
                norain = norain.unsqueeze(0)

            h,w = rain.shape[2], rain.shape[3]
            H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            rain = F.pad(rain, (0,padw,0,padh), 'reflect')
            norain = F.pad(norain, (0, padw, 0, padh), 'reflect')

            # Load Prompt #############################################
            # with torch.no_grad():
                # prompt =learn_prompt.get_text_embedding()
                # prompt = torch.ones_like(_prompt)

                # spatial prompt
                # spatial_feature =learn_prompt.get_img_embedding(spatial_features)
            #     prompt = prompt * label.unsqueeze(2)
            #     prompt = prompt.sum(dim=1, keepdim=True)

            # # clip & learnable prompt
            # output, prompt =learn_prompt(rain_feature,0)
            # _prompt = prompt[1:][:]
            # _output = output[:, 1:]

            # clean_prompt = prompt[0][:].unsqueeze(0)
            # clean_prompt = clean_prompt.repeat((output.shape[0]),1,1)

            # prompt = _prompt.repeat((output.shape[0]),1,1) * _output.softmax(dim=-1).unsqueeze(2) # if multi-prompt

            # _output_softmaxed = _output.softmax(dim=-1)
            # max_indices = torch.argmax(_output_softmaxed, dim=-1)
            # _output = torch.zeros_like(_output_softmaxed)
            # _output.scatter_(-1, max_indices.unsqueeze(-1), 1)
            # prompt = _prompt.repeat((output.shape[0]),1,1) * _output.unsqueeze(2)
            # prompt = prompt.sum(dim=1, keepdim=True)

            # prompt = torch.cat((clean_prompt,prompt), dim=1).cuda()

            # Model
            # out = torch.clamp((torch.clamp(model(rain)[:, :, :h, :w], 0, 1).mul(255)), 0, 255).byte()
            out = net(rain,norain)

            out = torch.clamp((torch.clamp(out[:, :, :h, :w], 0, 1).mul(255)), 0, 255)#.byte()

            # 获取颜色映射
            ###################################################

            # # 輸出degraded map 将张量的值缩放到 [0, 1] 的范围内
            # tensor_min = degra_map.min()
            # tensor_max = degra_map.max()
            # degra_map = (degra_map - tensor_min) / (tensor_max - tensor_min)
            # degra_map = torch.clamp((degra_map[:,:, :h, :w].mul(255)), 0, 255)

            # for i in range(degra_map.shape[1]):
            #     _degra_map = degra_map[:,i, :h, :w]
            #     # 輸出degraded map 将张量的值缩放到 [0, 1] 的范围内
            #     tensor_min = _degra_map.min()
            #     tensor_max = _degra_map.max()
            #     _degra_map = (_degra_map - tensor_min) / (tensor_max - tensor_min)
            #     _degra_map = torch.clamp((_degra_map.mul(255)), 0, 255)

            #     save_path = '{}/{}/{}'.format(args.save_path, data_name, name[0].split('.')[-2]+'_'+ str(i)+'.png')
            #     if not os.path.exists(os.path.dirname(save_path)):
            #         os.makedirs(os.path.dirname(save_path))
            #     # Image.fromarray(degra_map[:,i, :h, :w].squeeze().byte().contiguous().cpu().numpy()).save(save_path)

            #     # 创建热力图
            #     cmap = plt.get_cmap('jet')

            #     fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))

            #     im1 = ax1.imshow(_degra_map.squeeze().byte().contiguous().cpu().numpy(), cmap=cmap)
            #     ax1.set_title('Channel Normalized')

            #     # im2 = ax2.imshow(degra_map[:,i, :h, :w].squeeze().byte().contiguous().cpu().numpy(), cmap=cmap)
            #     # ax2.set_title('Channel_h Normalized')

            #     # 使用 make_axes_locatable 创建独立的轴
            #     divider1 = make_axes_locatable(ax1)
            #     cax1 = divider1.append_axes("right", size="5%", pad=0.05)
            #     plt.colorbar(im1, cax=cax1)

            #     # divider2 = make_axes_locatable(ax2)
            #     # cax2 = divider2.append_axes("right", size="5%", pad=0.05)
            #     # plt.colorbar(im2, cax=cax2)

            #     # 保存图像到指定文件夹，使用不同的文件名
            #     plt.savefig(save_path)

            #     # 关闭当前图，以便下一个循环迭代可以创建新的图
            #     plt.close()
            ###################################################


            norain = torch.clamp(norain[:, :, :h, :w].mul(255), 0, 255)#.byte()
            # computer the metrics with Y channel and double precision
            # y, gt = rgb_to_y(out.double()), rgb_to_y(norain.double())
            # current_psnr, current_ssim = psnr(y, gt), ssim(y, gt)
            # total_psnr += current_psnr#.item()
            # total_ssim += current_ssim.item()
            count += 1
            save_path = '{}/{}/{}'.format(args.save_path, data_name, name[0].replace('jpg','png'))
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            # ## Save #######################################\
            Image.fromarray(out.squeeze(0).permute(1, 2, 0).squeeze().byte().contiguous().cpu().numpy()).save(save_path)   # save the image
            # Image.fromarray(degra_map1.squeeze().byte().contiguous().cpu().numpy()).save(save_path)   # save the image
            test_bar.set_description(str(data_name)+' Test Iter: [{}/{}] count: {}'
                                     .format(num_iter, 1 if args.model_file else args.num_iter,count))

            # recoverd = out.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()
            # clean = norain.squeeze(dim=0).permute(1, 2, 0).contiguous().cpu().numpy()
            # psnr_rgb = peak_signal_noise_ratio(recoverd, clean,data_range=255)
            # ssim_rgb = structural_similarity(recoverd,clean, data_range=255, channel_axis=-1)
            # psnr_rgbs += psnr_rgb#.item()
            # ssim_rgbs += ssim_rgb
            # print(psnr_rgb.item(), "__", count)
        # print(f'Final : Avg PSNR_y {total_psnr / count}, Avg SSIM_y {total_ssim / count}')
        # print(f'avg psnr_rgb : {psnr_rgbs / count} , avg ssim_rgb : {ssim_rgbs / count}')

    return


def save_loop(net,data_loader, num_iter, data_name, save=True,mode=None):
    # global best_score
    # score = test_loop(net, data_loader, num_iter ,data_name)
    # results['PSNR'].append('{:.2f}'.format(val_psnr))
    # results['SSIM'].append('{:.3f}'.format(val_ssim))
    # if mode !='test':
    #     writer.add_scalars('Test score', {str(data_name)+"_score": score}, num_iter)
    # save statistics
    # data_frame = pd.DataFrame(data=results, index=range(1, (num_iter if args.stage=='test' else num_iter // args.val_iter) + 1))
    # data_frame.to_csv('{}/{}.csv'.format(args.save_path, data_name), index_label='Iter', float_format='%.3f')


        # if val_psnr > best_psnr_derain and val_ssim > best_ssim_derain:
        # if score > best_score :
            # best_score = score
            # with open('{}/{}.txt'.format(args.save_path, data_name), 'w') as f:
            #     f.write('Iter: {} PSNR:{:.2f}'.format(num_iter, best_score))
            # if save:
            #     # 保存模型狀態
            #     checkpoint = {
            #         'epoch': num_iter + 1,
            #         'model_state_dict': net.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #     }
            #     torch.save(checkpoint, '{}/{}.pth'.format(args.save_path, data_name))

    if save:
            # 保存模型狀態
            checkpoint = {
                'epoch': num_iter + 1,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, '{}/{}.pth'.format(args.save_path, 'All'))

        # if data_name == 'Test1':
        #     # if val_psnr > best_psnr_derain and val_ssim > best_ssim_derain:
        #     if val_psnr_y > best_psnr_derain :
        #         best_psnr_derain, best_ssim_derain = val_psnr_y, val_ssim_y
        #         with open('{}/{}.txt'.format(args.save_path, data_name), 'w') as f:
        #             f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr_derain, best_ssim_derain))
        #         if save:
        #             # 保存模型狀態
        #             checkpoint = {
        #                 'epoch': num_iter + 1,
        #                 'model_state_dict': net.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #             }
        #             torch.save(checkpoint, '{}/{}.pth'.format(args.save_path, data_name))
        #             # torch.save(net.state_dict(), '{}/{}.pth'.format(args.save_path, data_name))
        # elif data_name == 'RainDrop':
        #     # if val_psnr > best_psnr_deRainDrop and val_ssim > best_ssim_deRainDrop:
        #     if val_psnr_y > best_psnr_deRainDrop :
        #         best_psnr_deRainDrop, best_ssim_deRainDrop = val_psnr_y, val_ssim_y
        #         with open('{}/{}.txt'.format(args.save_path, data_name), 'w') as f:
        #             f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr_deRainDrop, best_ssim_deRainDrop))
        #         if save:
        #             # 保存模型狀態
        #             checkpoint = {
        #                 'epoch': num_iter + 1,
        #                 'model_state_dict': net.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #             }
        #             torch.save(checkpoint, '{}/{}.pth'.format(args.save_path, data_name))
        #             # torch.save(net.state_dict(), '{}/{}.pth'.format(args.save_path, data_name))
        # elif data_name == 'Snow':
        #     # if val_psnr > best_psnr_desnow and val_ssim > best_ssim_desnow:
        #     if val_psnr_y > best_psnr_desnow :
        #         best_psnr_desnow, best_ssim_desnow = val_psnr_y, val_ssim_y
        #         with open('{}/{}.txt'.format(args.save_path, data_name), 'w') as f:
        #             f.write('Iter: {} PSNR:{:.2f} SSIM:{:.3f}'.format(num_iter, best_psnr_desnow, best_ssim_desnow))
        #         if save:
        #             # 保存模型狀態
        #             checkpoint = {
        #                 'epoch': num_iter + 1,
        #                 'model_state_dict': net.state_dict(),
        #                 'optimizer_state_dict': optimizer.state_dict(),
        #             }
        #             torch.save(checkpoint, '{}/{}.pth'.format(args.save_path, data_name))
        #             # torch.save(net.state_dict(), '{}/{}.pth'.format(args.save_path, data_name))



if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    args = parse_args()
    writer = SummaryWriter('.\\'+args.save_path+"\\"+'tensorboard')
    # test_dataset_RainDrop = RainDataset(args, test_path ='.\\data\\allweather\\Test\\RainDrop')
    # test_dataset_Snow = RainDataset(args,test_path ='.\\data\\allweather\\Test\\Snow100K_L_all')
    test_dataset_Public = RainDataset(args,test_path ='.\\data\\Public_dataset')


    # test_loader_RainDrop = DataLoader(test_dataset_RainDrop, batch_size=1, shuffle=False, num_workers=args.workers,pin_memory=True)
    # test_loader_Snow = DataLoader(test_dataset_Snow, batch_size=1, shuffle=False, num_workers=args.workers,pin_memory=True)
    test_loader_Public = DataLoader(test_dataset_Public, batch_size=1, shuffle=False, num_workers=args.workers,pin_memory=True)


    best_score = 0.0
    model = Restormer(args.num_blocks, args.num_heads, args.channels, args.num_refinement, args.expansion_factor).cuda()
    # model = segnet().cuda()
    get_parameter_number(model)
    # L_clip_MSE = L_clip_MSE().cuda()

    # # Load Prompts
    # print("load pretrain prompt")
    # prompt_pretrain_dir = ".\\CLIP_LIT\\ViT-L_allweather_withGT2\\snapshots_prompt_ViT-L_allweather_withGT2\\best_prompt_round0.pth"
    # learn_prompt = Prompts_L(prompt_pretrain_dir, 16).cuda()
    # for para in learn_prompt.parameters():
    #     para.requires_grad = False
    # learn_prompt.eval()

    if args.stage == 'test':
        print("Testing ...")
        model.load_state_dict(torch.load(args.model_file)['model_state_dict'], strict=True)
        # model.load_state_dict(torch.load(args.model_file), strict=True)
        get_parameter_number(model)
        save_loop(model,test_loader_Public, 1,'Public', save=False ,mode=args.stage)
    else:
        if args.model_file:
            print(f'Load model ckpt from : {args.model_file} ....')
            model.load_state_dict(torch.load(args.model_file)['model_state_dict'], strict=True)
            # for para in model.parameters():
            #     para.requires_grad = False
            # model.eval()

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # optimizer.load_state_dict(torch.load(args.model_file)['optimizer_state_dict']) # resume
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_iter, eta_min=1e-6)
        # for k in range(240000):
        #     lr_scheduler.step()

        total_loss, total_num, i = 0.0, 0, 0 # resume i=4
        total_loss_sam = 0.0
        total_loss_mse = 0.0
        train_bar = tqdm(range(1, args.num_iter + 1), initial=1, dynamic_ncols=True) # resume iter
        for n_iter in train_bar:
            # progressive learning
            if n_iter == 1 or n_iter - 1 in args.milestone:
                end_iter = args.milestone[i] if i < len(args.milestone) else args.num_iter
                start_iter = args.milestone[i - 1] if i > 0 else 0
                length = args.batch_size[i] * (end_iter - start_iter)
                train_dataset = RainDataset(args,'train',args.patch_size[i],length)
                train_loader = iter(DataLoader(train_dataset, args.batch_size[i], shuffle=True, num_workers=args.workers,pin_memory=True))
                i += 1

            # train
            model.train()

            # rain, norain, name,label = next(train_loader)

            rain, norain, name = next(train_loader)

            rain, norain = rain.cuda(), norain.cuda()

            # with torch.no_grad():
                # text prompt
                # prompt =learn_prompt.get_text_embedding()
                # prompt = torch.ones_like(_prompt)

                # spatial prompt
                # spatial_feature =learn_prompt.get_img_embedding(spatial_features)

                # prompt = prompt * label.unsqueeze(2)
                # prompt = prompt.sum(dim=1, keepdim=True)

            # with torch.no_grad():
                # output, prompt =learn_prompt(rain_feature,0)
                # _prompt = prompt[1:][:]
                # _output = output[:, 1:]
                # clean_prompt = prompt[0][:].unsqueeze(0)
                # clean_prompt = clean_prompt.repeat((output.shape[0]),1,1)

                # # prompt = _prompt.repeat((output.shape[0]),1,1) * _output.softmax(dim=-1).unsqueeze(2) # if multi-prompt

                # _output_softmaxed = _output.softmax(dim=-1)
                # max_indices = torch.argmax(_output_softmaxed, dim=-1)
                # _output = torch.zeros_like(_output_softmaxed)
                # _output.scatter_(-1, max_indices.unsqueeze(-1), 1)

                # prompt = _prompt.repeat((output.shape[0]),1,1) * _output.unsqueeze(2)
                # prompt = prompt.sum(dim=1, keepdim=True)



                # prompt = torch.cat((clean_prompt,prompt), dim=1).cuda()


            out = model(rain,norain)

            # clip_MSEloss = L_clip_MSE(out, norain,[1.0,1.0,1.0,1.0,0.5])
            # clip_loss = learn_prompt.get_clip_score_from_feature(out)
            # loss = F.l1_loss(out, norain) #+ 0.5 * loss_ctr + loss_SAM   #+ clip_loss #+ clip_MSEloss

            loss = F.mse_loss(out, norain)
#GrayLoss(out) +
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_num += rain.size(0)
            total_loss += loss.detach().item() * rain.size(0)
            # total_loss_sam += loss_SAM.detach().item()* rain.size(0)
            # total_loss_mse += loss_ctr.detach().item()* rain.size(0)
            # train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f} Loss_sam {:.3f} Loss_ctr {:.3f}'
                                    #   .format(n_iter, args.num_iter, total_loss / total_num , total_loss_sam / total_num ,total_loss_mse / total_num))
            # train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f} Loss_ctr {:.3f}'
            #                           .format(n_iter, args.num_iter, total_loss / total_num , total_loss_sam / total_num))
            train_bar.set_description('Train Iter: [{}/{}] Loss: {:.3f}'
                                      .format(n_iter, args.num_iter, total_loss / total_num))

            lr_scheduler.step()
            if n_iter % args.val_iter == 0 :
                # writer.add_scalars('Train', {"loss": (total_loss / total_num),"loss_sam": (total_loss_sam / total_num),"loss_ctr": (total_loss_mse / total_num)}, n_iter)
                # writer.add_scalars('Train', {"loss": (total_loss / total_num),"loss_ctr": (total_loss_sam / total_num)}, n_iter)
                writer.add_scalars('Train', {"loss": (total_loss / total_num)}, n_iter)
                if n_iter % 2500 ==0 and n_iter< 250000:
                    print('='*100)
                    save_loop(model, test_loader_Public, n_iter,'All')
                    print('='*100)
                elif n_iter >= 250000 :
                    # results['Loss'].append('{:.3f}'.format(total_loss / total_num))
                    print('='*100)
                    save_loop(model, test_loader_Public, n_iter,'All')
                    print('='*100)
                    # save_loop(model, test_loader_RainDrop, n_iter, learn_prompt,'RainDrop')
                    # print()
                    # save_loop(model, test_loader_Snow, n_iter, learn_prompt,'Snow')
                    # print()
                    # save_loop(model, test_loader_Rain, n_iter, learn_prompt,'Test1')



'''
第272,273行可以修改要test的資料夾，第295行選擇測試的dataloader
python train.py --stage 'test' --model_file './result_AICUP_ori/All.pth' --save_path 'result_AICUP_ori'
'''