import os
import numpy as np
import argparse
import tensorflow as tf
from model_DnCNN import DnCNN

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Tensorflow DnCNN Train")
parser.add_argument("--epochs", default=50, type=int, help="Train epochs")
parser.add_argument("--patch-size", default=40, type=int, help="patch size")
parser.add_argument("--c-dim", default=1, type=int, help="# of channels")
parser.add_argument("--batch-size", default=64, type=int, help="mini-batch size")
parser.add_argument("--lr", default=1e-3, type=float, help="Learning rate")
parser.add_argument("--lr-decay", default=30, type=int, help="Step of learning rate decay")
parser.add_argument("--weight-decay", default=1e-4, type=float, help="Weight decay")
parser.add_argument("--sigma", default=25, type=int, help="noise level (default 25)")
parser.add_argument('--train-path', default='./data/imdb_40_128_V1.tfrecords', type=str, help='path to trainset')
parser.add_argument('--validate-dir', default='E:/denoising_data/_noisyset/', type=str, help='path to trainset')
parser.add_argument('--validate-set', default='Set12', type=str, help='path to trainset')
parser.add_argument('--model-name', default='DnCNN', type=str, help='path to checkpoints')
#parser.add_argument('--checkpoint-path', default='./DnCNN_checkpoints/', type=str, help='path to checkpoints')

opt = parser.parse_args()
opt.checkpoint_path = './'+ opt.model_name + '_checkpoints/'
def main(_):
    model = DnCNN()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model.train(sess, opt)

if __name__ == '__main__':
    tf.app.run()
