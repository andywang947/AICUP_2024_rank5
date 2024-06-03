import torch
import torch.nn as nn
import torch.nn.functional as F

class UpSample(nn.Module):
    def __init__(self, channels, scale=2.0):
        super(UpSample, self).__init__()
        self.scale = scale
        self.body = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),)
                                #   nn.PixelShuffle(scale))

    def forward(self, x):
        y = self.body(x)
        y = nn.functional.interpolate(y, scale_factor=self.scale, mode="nearest")
        return y
    
    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ResBlock, self).__init__()
        self.norm1 = nn.LayerNorm(in_ch)
        self.act1 = nn.GELU()
        self.conv1 = torch.nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1)

        self.norm2 = nn.LayerNorm(out_ch)
        self.act2 = nn.GELU()
        self.conv2 = torch.nn.Conv2d(out_ch,out_ch,kernel_size=1)

        self.conv_shortcut = torch.nn.Conv2d(in_ch,out_ch,kernel_size=3,stride=1,padding=1)
        
    def forward(self, x):
        h = x
        # h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)

        # h = self.norm2(h)
        h = self.act2(h)
        h = self.conv2(h)

        return h + self.conv_shortcut(x)
    

    
# class SPNet(nn.Module):
#     def __init__(self, in_ch, inner_ch=[384,192,96], num_heads=4, expansion_factor=2.66):
#         super(SPNet, self).__init__()
        
#         # input size : 7x7
#         # self.init_layer = nn.Conv2d(in_ch, inner_ch[0], 1)
#         self.layer1 = TransformerBlock(in_ch, num_heads, expansion_factor)
#         self.layer2 = TransformerBlock(in_ch, num_heads, expansion_factor)
#         self.layer3 = TransformerBlock(in_ch, num_heads, expansion_factor)
#         self.out = nn.Conv2d(in_ch, in_ch, 1)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.out(x)
#         return x

class SPNet(nn.Module):
    def __init__(self, in_ch, inner_ch):
        super(SPNet, self).__init__()

        # input size : 7x7
        self.init_layer = nn.Sequential(ResBlock(in_ch, inner_ch))
        self.layer1 = nn.Sequential(ResBlock(inner_ch, inner_ch))
        # self.up1 = UpSample(inner_ch[0])    # 7x7 -> 14x14 // (ch / 2)
        # self.layer2 = nn.Sequential(ResBlock(inner_ch, inner_ch))
        # self.up2 = UpSample(inner_ch[1])    # 14x14 -> 28x28 // (ch / 2)
        # self.layer3 = nn.Sequential(ResBlock(inner_ch, inner_ch))
        # # self.out = nn.Conv2d(inner_ch[2], inner_ch[2], 1)

    def forward(self, x):
        x = self.init_layer(x)
        x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        return x[:,:3,:,:]