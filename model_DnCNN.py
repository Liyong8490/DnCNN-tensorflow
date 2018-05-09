import os
import time
import scipy
import scipy.misc
import scipy.io as sio
import numpy as np
import tensorflow as tf
from tensorflow.python.training.moving_averages import assign_moving_average
import matplotlib.pyplot as plt
from utils import data_aug, c_psnr, c_ssim

class DnCNN(object):
    """docstring for DnCNN"""
    def __init__(self):
        super(DnCNN, self).__init__()

    def _build_model(self, lamb=0):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        results = self._build_dncnn(lamb=lamb, num_layers=17)
        return results

    def _conv2d(self, input, filter_shape, strides=[1,1,1,1], padding='SAME', baise=True, name=None):
        with tf.variable_scope(name, default_name='conv_noname') as scope:
            W = tf.get_variable('weights', filter_shape, initializer= \
                tf.constant_initializer((2 / (9.0 * 64)) ** 0.5 * self.sess.run(tf.truncated_normal(filter_shape))))
            if baise:
                b = tf.get_variable('biases', [1, filter_shape[-1]], initializer= \
                                    tf.constant_initializer(0))
                conv = tf.nn.conv2d(input, W, strides=strides, padding=padding) + b
            else:
                conv = tf.nn.conv2d(input, W, strides=strides, padding=padding)
        return conv

    def _conv2d_bn_relu(self, input, filter_shape, strides=[1,1,1,1], padding='SAME', name=None):
        conv = self._conv2d(input, filter_shape, strides=strides, padding=padding, baise=False, name=name)
        bn_name = None
        if name is not None: bn_name = name + '_bn'
        bn = tf.layers.batch_normalization(conv, training=self.is_training, name=bn_name)
        out = tf.nn.relu(bn)
        # with tf.variable_scope(name+'_bn'):
        #     out = tf.contrib.layers.batch_norm(conv, scale=True, is_training=self.is_training)
        return out

    def _build_dncnn(self, lamb=0, num_layers=17):
        fmsz = 64
        ksz = 3
        out = self._conv2d(self.inputs, [ksz, ksz, self.c_dim, fmsz], name='conv1')
        for i in range(2, num_layers):
            out = self._conv2d_bn_relu(out, [ksz, ksz, fmsz, fmsz], name='conv%d' %i)
            # out = self._conv2d(out, [ksz, ksz, fmsz, fmsz], name='conv%d' %i)
        results = self._conv2d(out, [ksz, ksz, fmsz, self.c_dim], name='results')
        return results

    def _build_dncnn_old(self, lamb=0, num_layers=17):
        kernelsize = (3,3)
        featuremap = 64
        weight_initial = (2 / (9.0 * featuremap)) ** 0.5
        with tf.variable_scope('conv1') as scope:
            out = tf.layers.conv2d(self.inputs, featuremap, kernelsize, padding='SAME', activation=tf.nn.relu,
                                   use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=weight_initial))
            # tf.add_to_collection('loss', tf.contrib.layers.l2_regularizer(lamb)(out.weight))
        for i in range(2, num_layers):
            with tf.variable_scope('conv%d' %i) as scope:
                conv = tf.layers.conv2d(out, featuremap, kernelsize, padding='SAME', name='conv%d'%i,
                                        use_bias=False, kernel_initializer=tf.truncated_normal_initializer(stddev=weight_initial))
                out = tf.nn.relu(tf.contrib.layers.batch_norm(conv, scale=True, is_training=self.is_training))
                # out = tf.layers.conv2d(out, featuremap, kernelsize, padding='SAME', name='conv%d'%i, activation=tf.nn.relu,\
                #                        use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=weight_initial))
        with tf.variable_scope('conv%d'%num_layers) as scope:
            results = tf.layers.conv2d(out, self.c_dim, kernelsize, padding='SAME',
                                       use_bias=True, kernel_initializer=tf.truncated_normal_initializer(stddev=weight_initial))
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
        self.c_dim = opt.c_dim
        # build network(s)
        self.inputs = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='inputs')
        self.labels = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='labels')

        self.results = self._build_model(lamb=weight_decay)
        self.lr = tf.placeholder(tf.float32, name='learning_rate') # to add decay
        # self.loss = (0.5 / batch_size) * tf.nn.l2_loss(self.results - self.labels)
        self.loss = (0.5 / batch_size) * tf.reduce_sum(tf.square(self.results - self.labels))
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
        # validate related
        validate_set = opt.validate_set
        validate_dir = opt.validate_dir
        test_datas = []
        test_labels = []
        test_names = []
        path = validate_dir + str(sigma) + '/' + validate_set  # file folder
        files = os.listdir(path)
        for file in files:
            imageName = os.path.splitext(file)[0]
            mat = scipy.io.loadmat(path + "/" + imageName + ".mat")
            original = mat['img']
            input_ = mat['noisyimg']
            original = original.astype(np.float32)
            input_ = input_.astype(np.float32)
            input_ = input_ / 255.0
            label_ = original / 255.0
            test_datas.append(input_)
            test_labels.append(label_)
            test_names.append(imageName)
        psnr_summary = []
        ssim_summary = []

        # training
        flag = False
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
        for epoch in range(1, nEpoch+1):
            dataset = tf.data.TFRecordDataset([train_path], num_parallel_reads=4)\
                        .map(datamap, num_parallel_calls=batch_size)\
                        .shuffle(buffer_size=batch_size*4*patch_size**2, reshuffle_each_iteration=True)\
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
                    mini_batch = np.array(inputs_val, dtype=np.float32)/255
                    mini_batch = np.reshape(mini_batch, [mini_batch.shape[0], patch_size, patch_size, 1])
                    rnd_aug = np.random.randint(8,size=mini_batch.shape[0])
                    for i in range(mini_batch.shape[0]):
                        mini_batch[i,:,:,:] = np.reshape(data_aug(
                            np.reshape(mini_batch[i,:,:,:], [patch_size,patch_size]),
                            rnd_aug[i]),[1, patch_size, patch_size, 1])
                    label_b = sigma / 255.0 * np.random.normal(size=np.shape(mini_batch))
                    input_b = mini_batch + label_b

                    if epoch < lr_decay:
                        _, loss, result = self.sess.run([self.train_op, self.loss, self.results], feed_dict={
                                                self.inputs:input_b, self.labels: label_b,
                                                self.lr:lr, self.is_training: True})
                    else:
                        _, loss, result = self.sess.run([self.train_op, self.loss, self.results], feed_dict={
                                                self.inputs:input_b, self.labels: label_b,
                                                self.lr:lr/10, self.is_training: True})

                    if flag:
                        for ind in range(batch_size):
                            scipy.misc.imsave('./tmp/'+str(ind)+'_in.png', np.squeeze(input_b[ind,:,:,0]))
                            scipy.misc.imsave('./tmp/'+str(ind)+'_resi_l.png', np.squeeze(label_b[ind,:,:,0]))
                            scipy.misc.imsave('./tmp/'+str(ind)+'_resi.png', np.squeeze(result[ind,:,:,0]))
                            scipy.misc.imsave('./tmp/'+str(ind)+'_label.png', (np.squeeze(input_b[ind,:,:,0]) - np.squeeze(label_b[ind,:,:,0])))
                            res = np.squeeze(input_b[ind,:,:,0]) - np.squeeze(result[ind,:,:,0])
                            res[res < 0] = 0
                            res[res > 1.] = 1.
                            scipy.misc.imsave('./tmp/'+str(ind)+'.png', (res))
                            flag = False
                    # print('iter: [%2d] Time: %4.2 Loss: %.6f\n' % (iter, time.time() - start_time, loss))
                    total_loss = total_loss + loss
                    step = step + 1
                    if step%50 == 0:
                        flag = True
                        losses.append(loss)
                        print("Epoch: [{}] Iterations: [{}] Time: {} Loss: {}".format(epoch, step, time.time() - start_time, loss))

            except tf.errors.OutOfRangeError:
                losses_aver.append(total_loss/step)
                print('Done training for %d epochs, %d steps. Time: %f, AverLoss: %f' % (epoch, step, time.time() - start_time, total_loss/step))

            checkpoint = opt.checkpoint_path
            if not os.path.exists(checkpoint):
                os.mkdir(checkpoint)
            print("[*] Saving model...{}".format(epoch))
            saver.save(self.sess, os.path.join(checkpoint, opt.model_name), global_step=epoch)
            psnr_aver, ssim_aver = self.validate(validate_set, test_datas, test_labels, test_names,epoch)
            psnr_summary.append(np.mean(psnr_aver))
            ssim_summary.append(np.mean(ssim_aver))
            print("model {}   PSNR={}   SSIM={}".format(epoch, np.mean(psnr_aver), np.mean(ssim_aver)))
        sio.savemat('DnCNN_'+validate_set+'_validate.mat', {'psnr': np.array(psnr_summary), 'ssim':np.array(ssim_summary)})

    def validate(self, validate_set, test_datas, test_labels, test_names, epoch):
        # load validate data
        out_path = './validate_DnCNN/'+validate_set+'/'+str(epoch)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        psnr_aver = []
        ssim_aver = []
        # for input_data, label_data, imageName in (test_datas, test_labels, test_names):
        for i in range(len(test_datas)):
            input_data = test_datas[i]
            label_data = test_labels[i]
            imageName = test_names[i]
            result = self.sess.run([self.results], feed_dict={
                                                    self.inputs:input_data[np.newaxis,:,:,np.newaxis],
                                                    self.is_training: False})
            result = np.squeeze(result)
            # residual = input_data - result
            residual = result
            # residual[residual < 0] = 0
            # residual[residual > 1.] = 1.
            scipy.misc.imsave(out_path+'/'+imageName+'_resi.png', residual)
            input_= input_data
            input_[input_ < 0] = 0
            input_[input_ > 1.] = 1.
            scipy.misc.imsave(out_path+'/'+imageName+'_in.png', input_)
            scipy.misc.imsave(out_path+'/'+imageName+'_label.png', label_data)
            result = input_data-result
            result[result < 0] = 0
            result[result > 1.] = 1.
            psnr_aver.append(c_psnr(result, label_data))
            ssim_aver.append(c_ssim(result, label_data))
            scipy.misc.imsave(out_path+'/'+imageName+'.png', result)
        return psnr_aver, ssim_aver

    def test(self, sess, opt):
        self.sess = sess
        sigma = opt.sigma
        model_start = opt.model_start
        model_stop = opt.model_stop
        checkpoint = opt.checkpoint_dir
        test_set = opt.test_set
        # test_set = 'trainset'
        test_dir = opt.test_dir
        self.c_dim = opt.c_dim

        self.inputs = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='inputs')
        # self.labels = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='labels')
        self.results = self._build_model()
        self.sess.run(tf.global_variables_initializer())
        # load test data
        test_datas = []
        test_labels = []
        test_names = []
        path = test_dir + str(sigma) + '/' + test_set  # file folder
        files = os.listdir(path)
        for file in files:
            imageName = os.path.splitext(file)[0]
            mat = scipy.io.loadmat(path + "/" + imageName + ".mat")
            original = mat['img']
            input_ = mat['noisyimg']
            original = original.astype(np.float32)
            input_ = input_.astype(np.float32)
            input_ = input_ / 255.0
            label_ = original / 255.0
            test_datas.append(input_)
            test_labels.append(label_)
            test_names.append(imageName)

        # validate
        # saver = tf.train.import_meta_graph(checkpoint+'/DnCNN-'+str(model_stop)+'.meta')
        saver = tf.train.Saver()
        psnr_summary = []
        ssim_summary = []

        for model_id in range(model_start,model_stop+1):
            # load pre-trained model
            print("[*] Reading checkpoint... [%d]" %model_id)
            saver.restore(self.sess, checkpoint+'/DnCNN-'+str(model_id))
            out_path = './results_DnCNN/'+test_set+'/'+str(model_id)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            psnr_aver = []
            ssim_aver = []
            # for input_data, label_data, imageName in (test_datas, test_labels, test_names):
            for i in range(len(test_datas)):
                input_data = test_datas[i]
                label_data = test_labels[i]
                imageName = test_names[i]
                result = self.sess.run([self.results], feed_dict={
                                                        self.inputs:input_data[np.newaxis,:,:,np.newaxis],
                                                        self.is_training: False})
                result = np.squeeze(result)
                # residual = input_data - result
                residual = result
                scipy.misc.imsave(out_path+'/'+imageName+'_resi.png', residual)
                input_= input_data
                input_[input_ < 0] = 0
                input_[input_ > 1.] = 1.
                scipy.misc.imsave(out_path+'/'+imageName+'_in.png', input_)
                scipy.misc.imsave(out_path+'/'+imageName+'_label.png', label_data)
                result = input_data-result
                result[result < 0] = 0
                result[result > 1.] = 1.
                psnr_aver.append(c_psnr(result, label_data))
                ssim_aver.append(c_ssim(result, label_data))
                scipy.misc.imsave(out_path+'/'+imageName+'.png', result)
            print("model {}   PSNR={}   SSIM={}".format(model_id, np.mean(psnr_aver), np.mean(ssim_aver)))
            psnr_summary.append(np.mean(psnr_aver))
            ssim_summary.append(np.mean(ssim_aver))
        sio.savemat('DnCNN_'+test_set+'_resArray.mat', {'psnr': np.array(psnr_summary), 'ssim':np.array(ssim_summary)})
        return

    def test_train(self, sess, opt):
        self.sess = sess
        sigma = opt.sigma
        model_start = opt.model_start
        model_stop = opt.model_stop
        checkpoint = opt.checkpoint_dir
        # test_set = opt.test_set
        test_set = 'trainset'
        test_dir = opt.test_dir
        self.c_dim = opt.c_dim

        self.inputs = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='inputs')
        # self.labels = tf.placeholder(tf.float32, [None, None, None, self.c_dim], name='labels')
        self.results = self._build_model()
        self.sess.run(tf.global_variables_initializer())
        # load test data
        # test_datas = []
        # test_labels = []
        # test_names = []
        # path = test_dir + str(sigma) + '/' + test_set  # file folder
        # files = os.listdir(path)
        # for file in files:
        #     imageName = os.path.splitext(file)[0]
        #     mat = scipy.io.loadmat(path + "/" + imageName + ".mat")
        #     original = mat['img']
        #     input_ = mat['noisyimg']
        #     original = original.astype(np.float32)
        #     input_ = input_.astype(np.float32)
        #     input_ = input_ / 255.0
        #     label_ = original / 255.0
        #     test_datas.append(input_)
        #     test_labels.append(label_)
        #     test_names.append(imageName)
        patch_size = 40
        batch_size = 64
        def datamap(record):
            keys_to_feature = {
                'inputs': tf.FixedLenFeature([], tf.string),
            }
            tf_features = tf.parse_single_example(record,features=keys_to_feature)
            inputs = tf.decode_raw(tf_features['inputs'], tf.uint8)
            inputs = tf.reshape(inputs, [patch_size, patch_size])
            return inputs
        dataset = tf.data.TFRecordDataset(['./data/imdb_40_128_V1.tfrecords'], num_parallel_reads=4)\
                    .map(datamap, num_parallel_calls=batch_size)\
                    .shuffle(buffer_size=batch_size*4*patch_size**2, reshuffle_each_iteration=True)\
                    .batch(batch_size)\
                    .repeat(1)
        iterator = dataset.make_one_shot_iterator()
        inputs_batch = iterator.get_next()
        inputs_val = self.sess.run(inputs_batch)
        mini_batch = np.array(inputs_val, dtype=np.float32)/255
        mini_batch = np.reshape(mini_batch, [mini_batch.shape[0], patch_size, patch_size, 1])
        rnd_aug = np.random.randint(8,size=mini_batch.shape[0])
        for i in range(mini_batch.shape[0]):
            mini_batch[i,:,:,:] = np.reshape(data_aug(
                np.reshape(mini_batch[i,:,:,:], [patch_size,patch_size]),
                rnd_aug[i]),[1, patch_size, patch_size, 1])
        label_b = sigma / 255.0 * np.random.normal(size=np.shape(mini_batch))
        input_b = mini_batch + label_b
        # validate
        # saver = tf.train.import_meta_graph(checkpoint+'/DnCNN-'+str(model_stop)+'.meta')
        saver = tf.train.Saver()
        psnr_summary = []
        ssim_summary = []

        for model_id in range(model_start,model_stop+1):
            # load pre-trained model
            print("[*] Reading checkpoint... [%d]" %model_id)
            saver.restore(self.sess, checkpoint+'/DnCNN-'+str(model_id))
            out_path = './results_DnCNN/'+test_set+'/'+str(model_id)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            psnr_aver = []
            ssim_aver = []
            for i in range(batch_size):
                input_data = np.squeeze(input_b[i, ...])
                label_data = np.squeeze(label_b[i, ...])
                imageName = str(i)
                result = self.sess.run([self.results], feed_dict={
                                                        self.inputs:input_data[np.newaxis,:,:, np.newaxis],
                                                        self.is_training: False})
                result = np.squeeze(result)
                # residual = input_data - result
                residual = result
                scipy.misc.imsave(out_path+'/'+imageName+'_resi.png', residual)
                input_= input_data
                input_[input_ < 0] = 0
                input_[input_ > 1.] = 1.
                scipy.misc.imsave(out_path+'/'+imageName+'_in.png', input_)
                scipy.misc.imsave(out_path+'/'+imageName+'_label.png', label_data)
                result = input_data-result
                result[result < 0] = 0
                result[result > 1.] = 1.
                psnr_aver.append(c_psnr(result, label_data))
                ssim_aver.append(c_ssim(result, label_data))
                scipy.misc.imsave(out_path+'/'+imageName+'.png', result)
            print("model {}   PSNR={}   SSIM={}".format(model_id, np.mean(psnr_aver), np.mean(ssim_aver)))
            psnr_summary.append(np.mean(psnr_aver))
            ssim_summary.append(np.mean(ssim_aver))
        sio.savemat('DnCNN_'+test_set+'_resArray.mat', {'psnr': np.array(psnr_summary), 'ssim':np.array(ssim_summary)})
        return
