import argparse
import os
import numpy as np
import shutil

from net.convnext import convnext_tiny
from net.convnext import convnext_small
from net.densenet import DenseNet
from net.ghostnet import ghostnet
from net.mobilenetv2 import MobileNetV2
from net.resnet import ResNet
from net.resnext import ResNeXt
from net.vgg import VGG

import torch
import torch.nn.functional as F
import torchvision.datasets as D
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader

from PIL import Image
from tqdm import tqdm


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default="LOG/Resnet20_Acc100.pt", type=str)
    parser.add_argument("--resize", default=50, type=int)
    parser.add_argument('--srcdir', default='E:\\00_Project\\2022\\DN2123601_Classification\\Data\\Labeled\\20220408C\\3', type=str)
    parser.add_argument('--outdir', default='test/outimages/20220408C_3_ResNet20_Acc100',
                        type=str)

    return parser.parse_args()


def main():
    opt = get_argparser()
    class_names = ['1', '2', '3', '4', '5', '6', '7', '8']
    if not os.path.exists(opt.outdir):
        os.makedirs(opt.outdir)

    device = torch.device('cuda')

    # model = ghostnet(num_classes=len(class_names))
    # model = MobileNetV2(num_classes=len(class_names))
    # model = MobileNetV3(num_classes=len(class_names))
    model = ResNet(depth=20, num_classes=len(class_names))
    # model = convnext_tiny(num_classes=len(class_names))
    # model = convnext_small(num_classes=len(class_names))

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.load_state_dict(torch.load(opt.checkpoint, map_location=device))
    model.eval()

    i = 0
    for root, dirs, files in os.walk(opt.srcdir):
        for file in files:
            src = os.path.join(opt.srcdir, file)
            img = Image.open(src).convert('RGB')

            # preprocess
            trans = T.Compose([
                T.Resize((opt.resize, opt.resize)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
            ])
            npimage = np.array(img)

            # predict
            with torch.no_grad():
                img = trans(img).unsqueeze(0)  # To tensor of NCHW
                img = img.to(device)
                preds = torch.softmax(model(img)[0], dim=-1).detach().cpu().numpy()

            # post process
            class_name = class_names[np.argmax(preds)]

            tar = os.path.join(opt.outdir, class_name, file)
            if not os.path.exists(os.path.join(opt.outdir, class_name)):
                os.makedirs(os.path.join(opt.outdir, class_name))
            shutil.copy(src, tar)
            i += 1
            print(f'[{i}/{len(files)}] {file} : {class_name}')


if __name__ == "__main__":
    main()