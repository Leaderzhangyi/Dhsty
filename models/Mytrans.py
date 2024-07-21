
import numpy as np
import torch.nn as nn
from models.transDecoder import TransformerDecoder,TransformerDecoderLayer
from timm.models.vision_transformer_hybrid import HybridEmbed
import warnings
warnings.filterwarnings("ignore")

# StyTr2
"""
Embedding       

Transformer 

Decoder 
"""

# ours
"""
Encoder 
+
TransDecoder 


Decoder 
"""

# input : 
class MytransDecoder(nn.Module):
    def __init__(self):
        super(MytransDecoder, self).__init__()
        d_model = 512
        decoder_norm = nn.LayerNorm(d_model)
        decoder_layer = TransformerDecoderLayer(d_model = d_model, nhead = 8, dim_feedforward = 2048, dropout = 0.1, activation = "relu", normalize_before = False)
        self.TransDecoder = TransformerDecoder(decoder_layer = decoder_layer,num_layers=3,norm = decoder_norm,return_intermediate = False)

    def forward(self, style, content):
        # torch.Size([4, 1024, 512])  b x n x c
        # import ipdb;ipdb.set_trace()
        style = style.permute(1, 0, 2)
        content = content.permute(1, 0, 2)

        # torch.Size([1024, 4 , 512]) h*w,b,c 

        # 计算hs？ 解码hs 得到Ics
        hs = self.TransDecoder(content,style,None,None,None)[0]
        N,B,C = hs.shape
        H = int(np.sqrt(N))
        hs = hs.permute(1, 2, 0).view(B,C,-1,H)
        return hs