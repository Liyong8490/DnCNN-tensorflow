import os
import globle
import numpy as np
import tensorflow as tf

class DnCNN(object):
	"""docstring for DnCNN"""
	def __init__(self, imsize, c_dim):
		super(DnCNN, self).__init__()
		self.imsize = imsize
		self.c_dim = c_dim
		return 0
	
    def _build_model(self)
    	results = self._build_dncnn()
    	return results

    def _build_dncnn(self, num_layers=17)
    	kernelsize = 3
    	featuremap = 64
        with tf.variable_scope('conv1') as scope:
        	out = tf.layers.conv2d(self.inputs, featuremap, kernelsize, padding='SAME', activation=tf.nn.relu, use_bias=True)
        for i in range(2, num_layers):
        	with tf.variable_scope('conv%d' %i) as scope:
        		conv = tf.layers.conv2d(out, featuremap, kernelsize, padding='SAME', name='conv%d'%i, use_bias=False)
        		out = tf.nn.relu(tf.nn.batch_normalization(conv, traning=self.is_training))
    	with tf.variable_scope('conv%d'%num_layers) as scope:
    		results = tf.layers.conv2d(out, featuremap, kernelsize, padding='SAME', use_bias=True)
    	return results

	def train(self, sess, opt)
		# set hyper-parameters
		self.sess = sess
		batchsize = opt.batchsize
		nEpoch = opt.nEpoch
		lr = opt.lr
		lr_decay = opt.lr_decay
		weight_decay = opt.weight_decay
		trainsetpath = opt.trainsetpath
		sigma = opt.sigma
		# build network(s)
		self.inputs = tf.placeholder(tf.float32, [None, self.imsize, self.imsize, self.c_dim], name='inputs')
    	self.labels = tf.placeholder(tf.float32, [None, self.imsize, self.imsize, self.c_dim], name='inputs')
    	self.is_training = tf.placeholder(tf.bool, name='is_training')
    	self.results = self._build_model()
    	self.lr = tf.placeholder(tf.float32, name='learning_rate') # to add decay
		self.loss = (1.0 / batch_size) * tf.nn.l2_loss(self.results - self.labels)
		self.optimizer = tf.train.AdamOptimizer(self.lr, name='AdamOptimizer')
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)
		# load data

		# train and record loss
		self.sess.run(tf.global_variables_initializer())

		return 0
	def validate(self, sess, opt)
		return