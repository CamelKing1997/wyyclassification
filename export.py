import os

import net.resnet
import onnx
import onnxsim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as D
import torchvision.transforms as T
from torch.utils.data.dataloader import DataLoader
from collections import OrderedDict
import logging
from sheen import Str, ColoredHandler
import argparse

logger = logging.getLogger("export")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(f'./LOG/exportlog.txt')
file_handler.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(ColoredHandler())


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", default="LOG/Resnet20_Acc100.pt", type=str)

    parser.add_argument("--num_classes", type=int, default=8)
    parser.add_argument("--input_shape", type=int, default=50)
    parser.add_argument("--save_path", type=str, default='onnx/')
    return parser.parse_args()


def main():
    opt = get_argparser()
    f = opt.save_path+os.path.basename(opt.checkpoint)[:-3]+'.onnx'
    w = h = opt.input_shape
    im = torch.rand((1, 3, h, w))

    model = net.resnet.ResNet(depth=20, num_classes=opt.num_classes)

    if opt.checkpoint != "":
        checkpoint = torch.load(opt.checkpoint)
        new_checkpoint = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:]  # remove `module.`
            new_checkpoint[name] = v
        model.load_state_dict(new_checkpoint)
        logger.info(f"Model restored from {opt.checkpoint}")
        del checkpoint

    torch.onnx.export(model, im, f, verbose=False, opset_version=13,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=['image'],
                      output_names=['output'])

    # Checks
    model_onnx = onnx.load(f)
    onnx.checker.check_model(model_onnx)
    logger.info(onnx.helper.printable_graph(model_onnx.graph))

    model_onnx, check = onnxsim.simplify(model_onnx)
    assert check, 'assert check failed'
    f = f[:-5] + '_sim.onnx'
    onnx.save(model_onnx, f)


if __name__ == "__main__":
    main()
