import os
import globle
import numpy as np
import tensorflow as tf

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch SANet Test")
parser.add_argument("--cuda", action="store_false", help="use cuda?")
parser.add_argument("--model", default=50, type=int, help="model path")
parser.add_argument("--model_s", default=1, type=int, help="model path")
# parser.add_argument("--model", default="model/model_ISSR_epoch_15.pth", type=str, help="model path")
parser.add_argument("--image", default="woman_GT", type=str, help="image name")
parser.add_argument("--scale", default=25, type=int, help="scale factor, Default: 4")
parser.add_argument('--dataset', default='../denoising_data/_noisyset/', type=str, help='path to general model')

opt = parser.parse_args()