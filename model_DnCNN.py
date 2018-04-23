import os
import time
import scipy.io as sio
import numpy as np
import tensorflow as tf
from utils import data_aug

class DnCNN(object):
    """docstring for DnCNN"""
    def __init__(self, imsize, c_dim):
        super(DnCNN, self).__init__()
        self.imsize = imsize
        self.c_dim = c_dim

    def __build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        results = self.__build_dncnn()
        return results

    def __build_dncnn(self, num_layers=17):
        kernelsize = 3
        featuremap = 64
        with tf.variable_scope('conv1') as scope:
            out = tf.layers.conv2d(self.inputs, featuremap, kernelsize, padding='SAME', activation=tf.nn.relu, use_bias=True)
        for i in range(2, num_layers):
            with tf.variable_scope('conv%d' %i) as scope:
                conv = tf.layers.conv2d(out, featuremap, kernelsize, padding='SAME', name='conv%d'%i, use_bias=False)
                out = tf.nn.relu(tf.layers.batch_normalization(conv, training=self.is_training))
        with tf.variable_scope('conv%d'%num_layers) as scope:
            results = tf.layers.conv2d(out, featuremap, kernelsize, padding='SAME', use_bias=True)
        return results

    def train(self, sess, opt):
        # set hyper-parameters
        self.sess = sess
        patch_size = opt.patch_size
        batch_size = opt.batch_size
        nEpoch = opt.epochs
        lr = opt.lr
        lr_decay = opt.lr_decay
        weight_decay = opt.weight_decay
        train_path = opt.train_path
        sigma = opt.sigma

        # build network(s)
        self.inputs = tf.placeholder(tf.float32, [None, self.imsize, self.imsize, self.c_dim], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, self.imsize, self.imsize, self.c_dim], name='inputs')

        self.results = self.__build_model()
        self.lr = tf.placeholder(tf.float32, name='learning_rate') # to add decay
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.results - self.labels)
        self.optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss)
        # load data
        filename_queue = tf.train.string_input_producer([train_path])
        reader = tf.TFRecordReader()
        _, serialized_examples = reader.read(filename_queue)
        tf_features = tf.parse_single_example(serialized_examples, features={'inputs': tf.FixedLenFeature([], tf.string)})
        inputs_raw = tf_features['inputs']
        inputs = tf.decode_raw(inputs_raw, tf.uint8)
        inputs_batch = tf.train.shuffle_batch(inputs, batch_size, capacity=5000, min_after_dequeue=1000, num_threads=4)
        # train and record loss
        self.sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=self.sess)
        start_time = time.time()
        for epoch in range(1, nEpoch+1):
            inputs_val = self.sess.run(inputs_batch)
            mini_batch = np.array(inputs_val, dtype=np.float32)
            rnd_aug = np.random.randint(8,size=mini_batch.shape[0])
            for i in range(mini_batch.shape[0]):
                mini_batch[i,:,:,:] = np.reshape(data_aug(
                    np.reshape(mini_batch[i,:,:,:], [patch_size,patch_size]),
                    rnd_aug[i]),[1, 1, patch_size, patch_size])
            label_b = sigma / 255.0 * np.random.normal(size=np.shape(mini_batch))
            input_b = mini_batch + label_b
            _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.inputs:input_b,
                                   self.labels: label_b, self.is_training: True})
            print('Epoch: [%2d] Time: %4.2 Loss: %.6f\n',epoch, time.time() - start_time, loss)
            if epoch%lr_decay == 0:
                saver = tf.train.Saver()
                checkpoint = opt.checkpoint_path
                if not os.path.exists(checkpoint):
                    os.mkdir(checkpoint)
                print("[*] Saving model...")
                saver.save(self.sess, os.path.join(checkpoint, opt.model_name), global_step=epoch)
		
    def validate(self, sess, opt):
        return
