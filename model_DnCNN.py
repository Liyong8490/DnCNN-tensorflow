import os
import time
import scipy.io as sio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
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
        self.labels = tf.placeholder(tf.float32, [None, self.imsize, self.imsize, self.c_dim], name='labels')

        self.results = self.__build_model()
        self.lr = tf.placeholder(tf.float32, name='learning_rate') # to add decay
        self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.results - self.labels)
        self.optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.loss)
        # load data

        if not os.path.exists(train_path):
            print("File: \"%s\" not found." % train_path)
            exit()
        # read train dataset from tfrecord
        def datamap(record):
            keys_to_feature = {
                'inputs': tf.FixedLenFeature([], tf.string),
            }
            tf_features = tf.parse_single_example(record,features=keys_to_feature)
            inputs = tf.decode_raw(tf_features['inputs'], tf.uint8)
            inputs = tf.reshape(inputs, [patch_size, patch_size])
            return inputs

        # filename_queue = tf.train.string_input_producer([train_path])
        # reader = tf.TFRecordReader()
        # _, serialized_examples = reader.read(filename_queue)
        # tf_features = tf.parse_single_example(serialized_examples,
        #                                       features={'inputs': tf.FixedLenFeature([], tf.string)
        #                                                 })
        # inputs = tf.decode_raw(tf_features['inputs'], tf.uint8)
        # inputs = tf.reshape(inputs, [patch_size, patch_size])
        # inputs_batch = tf.train.shuffle_batch([inputs], batch_size, capacity=5000, min_after_dequeue=1000, num_threads=4)
        # train and record loss
        self.sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners(sess=self.sess)
        losses = []
        losses_aver = []
        for epoch in range(1, nEpoch+1):
            dataset = tf.data.TFRecordDataset([train_path], num_parallel_reads=4)\
                        .map(datamap, num_parallel_calls=batch_size)\
                        .shuffle(buffer_size=batch_size*4*patch_size**2, seed=0, reshuffle_each_iteration=True)\
                        .batch(batch_size)\
                        .repeat(1)
            iterator = dataset.make_one_shot_iterator()
            inputs_batch = iterator.get_next()
            step = 0
            start_time = time.time()
            total_loss = 0

            try:
                step = 0
                while True:
                    inputs_val = self.sess.run(inputs_batch)
                    mini_batch = np.array(inputs_val, dtype=np.float32)
                    mini_batch = np.reshape(mini_batch, [mini_batch.shape[0], patch_size, patch_size, 1])
                    rnd_aug = np.random.randint(8,size=mini_batch.shape[0])
                    for i in range(mini_batch.shape[0]):
                        mini_batch[i,:,:] = np.reshape(data_aug(
                            np.reshape(mini_batch[i,:,:,:], [patch_size,patch_size]),
                            rnd_aug[i]),[1, patch_size, patch_size, 1])
                    label_b = sigma / 255.0 * np.random.normal(size=np.shape(mini_batch))
                    input_b = mini_batch / 255.0 + label_b
                    if epoch < lr_decay:
                        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
                                                self.inputs:input_b, self.labels: label_b,
                                                self.lr:lr, self.is_training: True})
                    else:
                        _, loss = self.sess.run([self.train_op, self.loss], feed_dict={
                                                self.inputs:input_b, self.labels: label_b,
                                                self.lr:lr/10, self.is_training: True})
                    # print('iter: [%2d] Time: %4.2 Loss: %.6f\n' % (iter, time.time() - start_time, loss))
                    total_loss = total_loss + loss
                    step = step + 1
                    if step%50 == 0:
                        losses.append(loss)
                        print("Epoch: [{}] Iterations: [{}] Time: {} Loss: {}".format(epoch, step, time.time() - start_time, loss))

            except tf.errors.OutOfRangeError:
                losses_aver.append(total_loss/step)
                print('Done training for %d epochs, %d steps. Time: %f, AverLoss: %f' % (epoch, step, time.time() - start_time, total_loss/step))

            saver = tf.train.Saver()
            checkpoint = opt.checkpoint_path
            if not os.path.exists(checkpoint):
                os.mkdir(checkpoint)
            print("[*] Saving model...{}".format(epoch))
            saver.save(self.sess, os.path.join(checkpoint, opt.model_name), global_step=epoch)
		
    def validate(self, sess, opt):
        return
