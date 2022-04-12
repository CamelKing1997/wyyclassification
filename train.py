import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as D
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader

import logging
from sheen import Str, ColoredHandler
import argparse

from net.convnext import convnext_tiny
from net.convnext import convnext_small
from net.densenet import DenseNet
from net.ghostnet import ghostnet
from net.mobilenetv2 import MobileNetV2
from net.resnet import ResNet
from net.resnext import ResNeXt
from net.vgg import VGG

logger = logging.getLogger("train")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(f'./LOG/trainlog.txt')
file_handler.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(ColoredHandler())


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default="LOG/ResNet8_E500_A99.99302455.pt", type=str)

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--resize", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lrstep", type=int, default=100)
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--Optimizer', type=str, default='SGD')

    parser.add_argument('--dataset', type=str, default='datasets/DN2123601/20220408CND')
    parser.add_argument('--val_interval', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=20)

    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=4096)
    parser.add_argument("--val_batch_size", type=int, default=4096)
    return parser.parse_args()


def main():
    opt = get_argparser()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    train_dataset = D.ImageFolder(
        os.path.join(opt.dataset, 'train'),
        T.Compose([
            T.Resize((opt.resize, opt.resize)),
            T.ColorJitter(0.3, 0.3, 0.3, 0.2),
            T.RandomHorizontalFlip(0.5),
            T.RandomVerticalFlip(0.5),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
    )
    train_dataloader = DataLoader(train_dataset, batch_size=opt.train_batch_size, pin_memory=True, shuffle=True, num_workers=opt.workers)
    val_dataset = D.ImageFolder(
        os.path.join(opt.dataset, 'val'),
        T.Compose([
            T.Resize((opt.resize, opt.resize)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ])
    )
    val_dataloader = DataLoader(val_dataset, batch_size=opt.val_batch_size,  pin_memory=True, shuffle=True, num_workers=opt.workers)

    logger.info(f"Train Batchsize:{opt.train_batch_size} Validation Batchsize:{opt.val_batch_size}.")
    logger.info(f"Dataset Loaded.")
    logger.info(f"Category:{len(train_dataset.classes)}.")
    logger.info(f'Train Image Count:{len(train_dataset.imgs)},Validation Image Count:{len(val_dataset.imgs)}.')
    logger.info(f'Train Iteration Count:{len(train_dataloader)},Validation Iteration Count:{len(val_dataloader)}.')

    # model = ghostnet(num_classes=len(train_dataset.classes))
    # modeltype = 'GhostNet'
    # model = MobileNetV2(num_classes=len(train_dataset.classes))
    # modeltype = 'MobileNetV2'
    # model = MobileNetV3(num_classes=len(train_dataset.classes))
    # modeltype = 'MobileNetV3'
    model = ResNet(depth=8, num_classes=len(train_dataset.classes))
    modeltype = 'ResNet8'
    # model = ResNet(depth=56, num_classes=len(train_dataset.classes))
    # modeltype = 'ResNet56'
    # model = convnext_tiny(num_classes=len(train_dataset.classes))
    # modeltype = 'ConvNeXt_Tiny'
    # model = convnext_small(num_classes=len(train_dataset.classes))
    # modeltype = 'ConvNeXt_Small'

    logger.info(f'{modeltype} Loaded.')

    torch.backends.cudnn.benchmark = True

    if opt.checkpoint != "":
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        model.load_state_dict(torch.load(opt.checkpoint, map_location=device))
        logger.info(f"Training state restored from {opt.checkpoint}")
    else:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        weights_init(model)

    logger.info(f'Model loaded into MultiGPU.')
    # device_count = torch.cuda.device_count()
    # if device_count > 1:
    #     model = torch.nn.DataParallel(model, device_ids=[0, 1])
    #     logger.info(f'Model loaded into MultiGPU.')
    # else:
    #     model.to(device)
    #     logger.info(f'Model loaded into GPU.')

    # Set Optimizer
    if opt.Optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)

    # Set Scheduler
    len_loader = (len(train_dataloader) * opt.epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda x: (1 - x / len_loader) ** 0.9)

    # Set Loss
    criterion = torch.nn.CrossEntropyLoss()

    c_epoch = 0
    while True:
        c_epoch += 1
        total_train_loss = 0
        total_train_accuracy = 0
        for iter, (image, label) in enumerate(train_dataloader):
            model.train()

            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            predict = model(image)
            train_loss = criterion(predict, label)
            train_loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += train_loss.item()

            train_accuracy = torch.mean(
                (torch.argmax(F.softmax(predict, dim=-1), dim=-1) == label).type(torch.FloatTensor))
            total_train_accuracy += train_accuracy

        meanloss = total_train_loss / (iter + 1)
        meanaccuracy = total_train_accuracy / (iter + 1)
        logger.info(
            f'Epoch:{c_epoch} TrainLoss:{meanloss:.05f} TrainAccuray:{meanaccuracy:.08f} lr:{get_lr(optimizer):.8f}')
        total_train_loss = 0
        total_train_accuracy = 0

        if c_epoch % opt.val_interval == 0:
            logger.info(f'Epoch:{c_epoch} Start Validation...')
            val_total_loss = 0
            val_total_accuracy = 0
            model.eval()
            for val_iteration, (image, label) in enumerate(val_dataloader):
                image = image.to(device)
                label = label.to(device)
                predict = model(image)
                val_loss = criterion(predict, label)
                val_total_loss += val_loss.item()

                val_accuracy = torch.mean(
                    (torch.argmax(F.softmax(predict, dim=-1), dim=-1) == label).type(torch.FloatTensor))
                val_total_accuracy += val_accuracy.item()
            val_meanloss = val_total_loss / (val_iteration + 1)
            val_meanaccuracy = val_total_accuracy / (val_iteration + 1)
            logger.info(f'Epoch:{c_epoch} ValidationLoss:{val_meanloss:.05f} ValidationAccuray:{val_meanaccuracy:.08f}')

        if c_epoch % opt.save_interval == 0:
            ptname = f'{modeltype}_E{c_epoch}_A{(val_meanaccuracy*100):.08f}.pt'
            torch.save(model.state_dict(), f'LOG/{ptname}')
            logger.info(f'Epoch:{c_epoch} Model save as \'LOG/{ptname}\'')

        if c_epoch == opt.epoch:
            break


def weights_init(net, init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    logger.info('Initialize network with kaiming type')
    net.apply(init_func)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == "__main__":
    main()
