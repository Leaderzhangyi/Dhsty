import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)
from function import normal,normal_style
from function import calc_mean_std
import scipy.stats as stats
from models.ViT_helper import DropPath, to_2tuple, trunc_normal_
from util.HSV import HSV
from util.misc import get_edge
import matplotlib.pyplot as plt 
from torchvision.utils import save_image
class DepthwiseConv(nn.Module):
    """
    深度可分离卷积层
    """
    def __init__(self, in_channels, kernel_size, stride=1, padding=1, bias=True):
        super(DepthwiseConv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding, groups=in_channels, bias=bias),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=bias,padding=0),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU()
        )

    def forward(self, inputs):
        outputs = self.layers(inputs)
        return outputs
    

class PatchEmbed(nn.Module):
    """ 
        Image to Patch Embedding
    """
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size


        # 保证起始与最终，中间层看自己造化
        # [4,3,256,256] - > [4,512,32,32]
        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # self.up1 = nn.Upsample(scale_factor=2, mode='nearest')

        # [4,512,64,64]
        self.conv1 = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


        part_size = embed_dim // 2

        print(f"part_size = {part_size}")
        self.dw1 = DepthwiseConv(in_channels = part_size, kernel_size=3, padding=1)
        self.dw2 = DepthwiseConv(in_channels = part_size, kernel_size=5, padding=2)
        self.c1 = nn.Conv2d(1024, 512, kernel_size=1)
        self.c2 = nn.Conv2d(1024, 512, kernel_size=1)

        self.avg1 = nn.MaxPool2d(kernel_size=3,stride=1, padding=1) 
        self.avg2 = nn.MaxPool2d(kernel_size=5,stride=1, padding=2)


        # self.layer1 = nn.Sequential (
        #     nn.AdaptiveMaxPool2d((32, 32)),
        #     nn.Conv2d(part_size, part_size, kernel_size=1),  # 使用1x1卷积替代线性层
        #     nn.GELU()

        # )
        # self.layer2 = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((32, 32)),
        #     nn.Conv2d(part_size, part_size, kernel_size=1),  # 使用1x1卷积替代线性层
        #     nn.GELU()
        # )





    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv1(x) # [4,3,256,256]
        # import ipdb; ipdb.set_trace()
        # xold_1 = x 
        # x1, x2 = torch.chunk(x, 2, dim=1)  # [4,512,64,64] -> 2 * [4,256,64,64]
        # x1 = self.dw1(x1)
        # x2 = self.dw2(x2)

        # x = torch.cat([x1,x2, xold_1],dim=1)  # [4,1024,32,32]
        # x = self.c1(x)
        # x_old2 = x
        # x1, x2 = torch.chunk(x, 2, dim=1)
        # x1 = self.avg1(x1)
        # x2 = self.avg2(x2)
        # # import pdb; pdb.set_trace()
        # x = torch.cat([x1,x2, x_old2],dim=1)
        # x = self.c2(x)
        return x

# 解码器中 9个填充
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class StyTrans(nn.Module):
    """ This is the style transform transformer module """
    
    def __init__(self,encoder,decoder,PatchEmbed, transformer,args):
        """
            encoder: vgg
            decoder: StyTR.decoder
            PatchEmbed: 作者自己设计的CAPE位置编码
            transformer: 多头注意力机制
        """
        # encoder 为vgg

        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        
        # 不需要梯度
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        # mse_loss 
        self.mse_loss = nn.MSELoss()
        self.transformer = transformer
        hidden_dim = transformer.d_model       
        self.decode = decoder
        self.embedding = PatchEmbed

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
      assert (input.size() == target.size())
      assert (target.requires_grad is False)
      return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)
    
    def forward(self, samples_c: NestedTensor,samples_s: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        """
        # batch_size,channel,height,width
        content_input = samples_c
        style_input = samples_s
        # [4 3 256 256]
        # print("content_input_1:",content_input.size())
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(samples_c)   # support different-sized images padding is used for mask [tensor, mask] 
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s) 

        # [4 3 256 256]
        # print("samples_c_2:",samples_c.tensors.size())
        
        # ### features used to calcate loss 
        content_feats = self.encode_with_intermediate(samples_c.tensors)
        style_feats = self.encode_with_intermediate(samples_s.tensors)
    

        # print("content_input_3:",content_feats.size())


        ### Linear projection
        style = self.embedding(samples_s.tensors)
        content = self.embedding(samples_c.tensors)

        # [4,512,32,32] b c h w
        # print("content_input_emedding_4:",content.size())

        
        # postional embedding is calculated in transformer.py
        pos_s = None
        pos_c = None

        mask = None
        hs = self.transformer(style, mask , content, pos_c, pos_s)  
        # torch.Size([4, 512, 32, 32])

        Ics = self.decode(hs)

        for i in range(Ics.shape[0]):
            save_image(Ics, f"./imgs/Ics_{i}.png")


        Ics_feats = self.encode_with_intermediate(Ics)

        # 计算编码后两层的损失
     
        loss_c = self.calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1]))
        + self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))


        # Style loss
        # 计算编码的每层损失差距 （感觉这里计算内容 和 风格损失借鉴了Gays的方法）
        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])
        
        # 输入自身图片，得到重建后的图片
        Icc = self.decode(self.transformer(content, mask , content, pos_c, pos_c))
        Iss = self.decode(self.transformer(style, mask , style, pos_s, pos_s))    

        # Identity losses lambda 1    
        # 一致性损失 
        loss_lambda1 = self.calc_content_loss(Icc,content_input)
        + self.calc_content_loss(Iss,style_input)
        
        # Identity losses lambda 2
        Icc_feats=self.encode_with_intermediate(Icc)
        Iss_feats=self.encode_with_intermediate(Iss)
        loss_lambda2 = self.calc_content_loss(Icc_feats[0], content_feats[0])+self.calc_content_loss(Iss_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_lambda2 += self.calc_content_loss(Icc_feats[i], content_feats[i])+self.calc_content_loss(Iss_feats[i], style_feats[i])
        # Please select and comment out one of the following two sentences

        # I_s [4,3,256,256]
        # I_cs [4,3,256,256]
        edgesList = []
        img_s = style_input.cpu().permute(0, 2, 3, 1).numpy()  # [4,256,256,3]
        for i in range(img_s.shape[0]):
            edgesList.append(get_edge(img_s[i]))
        Edge = torch.stack(edgesList)
        # print("Edge:",Edge.size())

        # 显示原始图片和提取的纹理
        # fig, axes = plt.subplots(4, 2, figsize=(12, 24))
        # for i in range(4):
        #     axes[i, 0].imshow(img_s[i])
        #     axes[i, 0].set_title(f'Original Image {i+1}')
        #     axes[i, 0].axis('off')
            
        #     axes[i, 1].imshow(Edge[i], cmap='gray')
        #     axes[i, 1].set_title(f'Canny Edges {i+1}')
        #     axes[i, 1].axis('off')
        # plt.savefig("./pp.png",dpi = 120)

        # print(f"sinput: {style_input.size()},Ics: {Ics.size()}, Edge: {Edge.size()}")
        # 需要颜色约束损失
        # 这里主要是 加深迁移图像的颜色 用迁移图 和 风格图做差
        # H 表示Hue 色调  S Saturation表示饱和度  V Value表示明度
       
        HSV_loss_H, HSV_loss_S, HSV_loss_V, HSV_loss = HSV(style_input, Ics, Edge)


        #Lambda_HSV: 10
        #Lambda_LHSV: 1
        LHSV = 20 * HSV_loss_H + 10 * HSV_loss_S



        return Ics,  loss_c, loss_s, loss_lambda1, loss_lambda2, LHSV   #train
        # return Ics,  LHSV   #train

    

        # return Ics    #test 