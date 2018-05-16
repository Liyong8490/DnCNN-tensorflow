import os
import numpy as np
import glob
import argparse
import tensorflow as tf
from model_DnCNN import DnCNN

parser = argparse.ArgumentParser(description='')
parser.add_argument("--c-dim", default=1, type=int, help="# of channels")
parser.add_argument('--model-start', dest='model_start', type=int, default=1, help='validate start epoch')
parser.add_argument('--model-stop', dest='model_stop', type=int, default=50, help='validate stop epoch')
parser.add_argument('--sigma', dest='sigma', type=int, default=25, help='noise level')
parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', default='./DnCNN_checkpoints', help='models are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='E:/denoising_data/_noisyset/', help='test sample are saved here')
parser.add_argument('--test_set', dest='test_set', default='BSD68', help='dataset for testing')
parser.add_argument('--gpus', dest='gpus', default='0', help='GPU ids, for example \'0,1\'')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

def main(_):
    model = DnCNN()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model.test(sess, args)

if __name__ == '__main__':
    tf.app.run()
