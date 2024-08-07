import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm
from pathlib import Path
from models.sampler import InfiniteSamplerWrapper
from models.VitaminEncoder import *
import models.transformer as transformer
import models.transDecoder as transDecoder
from models.Mytrans import MytransDecoder
import models.StyTR  as StyTR 
import timm
from torchvision.utils import save_image

from torchsummary import summary


# 图像预处理
def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256), 
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)             
        else:
            self.paths = list(Path(self.root).glob('*'))
        self.transform = transform
    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB') 
        img = self.transform(img)
        return img
    def __len__(self):
        return len(self.paths)
    def name(self):
        return 'FlatFolderDataset'

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='./datasets/train2014', type=str,   
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='./datasets/Images', type=str,  #wikiart dataset crawled from https://www.wikiart.org/
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./experiments/vgg_normalised.pth')  #run the train.py, please download the pretrained vgg checkpoint

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()


# cuda是否可用
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
print("device: ",device)
# if USE_CUDA == True:
#     print(torch.cuda.device_count())
#     for i in range(torch.cuda.device_count()):
#         deviceName = torch.cuda.get_device_name(i)
# else:
#     deviceName = "CPU"
# print("Using device: ",deviceName)

# 检查存储路径是否存在
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


# 检查日志文件是否存在
if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)



# 加载vgg
vgg = StyTR.vgg
vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])

# 加载transformer
decoder = StyTR.decoder
embedding = StyTR.PatchEmbed().to(device)
Trans = transformer.Transformer()
Mytrans = MytransDecoder().to(device)

# 加载Vitamin的Encoder
vitaminEncoder = timm.create_model('vitamin_base',pretrained=False).to(device)


# 各个模块拼接 组成整体网络
with torch.no_grad():
    network = StyTR.StyTrans(vgg,decoder,Mytrans,vitaminEncoder,args)
network.train()

network.to(device)

# cc = torch.rand(4, 3, 256, 256).to(device)
# ss = torch.rand(4, 3, 256, 256).to(device)
# summary(network,cc,ss)



# network = nn.DataParallel(network, device_ids=[0])
content_tf = train_transform()
style_tf = train_transform()


# 加载内容和图像数据集，运用自定义的图像transform，用自己构造的Dataset类来得到dataset，用于后续加载loader
content_dataset = FlatFolderDataset(args.content_dir, content_tf)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

# dataset -> dataloader
content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))



# 定义优化器
optimizer = torch.optim.Adam([ 
                            {'params': network.vitaEncoder.parameters()},
                              {'params': network.decode.parameters()},
                              {'params': network.TransDecoder.parameters()},        
                              ], lr=args.lr)

if not os.path.exists(args.save_dir+"/test"):
    os.makedirs(args.save_dir+"/test")


progress_bar = tqdm(range(args.max_iter),desc='Training')
for i in progress_bar:

    if i < 1e4:
        warmup_learning_rate(optimizer, iteration_count=i)
    else:
        adjust_learning_rate(optimizer, iteration_count=i)

    # print('learning_rate: %s' % str(optimizer.param_groups[0]['lr']))
    # 从迭代器中加载内容和风格图像
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)  
    # print(content_images.shape,style_images.shape)

    # 
    out, loss_c, loss_s,l_identity1, l_identity2, LHSV = network(content_images, style_images)
    # print(loss_c.sum().cpu().detach().numpy(),loss_s.sum().cpu().detach().numpy(),l_identity1.sum().cpu().detach().numpy(),l_identity2.sum().cpu().detach().numpy(),LHSV.sum().cpu().detach().numpy())
    if i % 100 == 0:
        output_name = '{:s}/test/{:s}{:s}'.format(
                        args.save_dir, str(i),".jpg"
                    )
        out = torch.cat((content_images,out),0)
        out = torch.cat((style_images,out),0)
        save_image(out, output_name)

        
    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    loss = loss_c + loss_s + (l_identity1 * 70) + (l_identity2 * 1)  + LHSV * 2
    

    # print(loss.sum().cpu().detach().numpy(),"-content:",loss_c.sum().cpu().detach().numpy(),"-style:",loss_s.sum().cpu().detach().numpy()
    #           ,"-l1:",l_identity1.sum().cpu().detach().numpy(),"-l2:",l_identity2.sum().cpu().detach().numpy()
    #           )
    progress_bar.set_postfix({'loss': loss.sum().cpu().detach().numpy()})
       
    optimizer.zero_grad()
    loss.sum().backward()
    optimizer.step()

    writer.add_scalar('loss_content', loss_c.sum().item(), i + 1)
    writer.add_scalar('loss_style', loss_s.sum().item(), i + 1)
    writer.add_scalar('loss_identity1', l_identity1.sum().item(), i + 1)
    writer.add_scalar('loss_identity2', l_identity2.sum().item(), i + 1)
    writer.add_scalar('total_loss', loss.sum().item(), i + 1)

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        state_dict = network.transformer.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/transformer_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

        state_dict = network.decode.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.embedding.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/embedding_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))

                                                    
writer.close()


