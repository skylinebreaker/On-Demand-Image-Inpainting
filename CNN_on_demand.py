import os
import sys
import numpy as np
import scipy.io
#import scipy.misc
import tensorflow as tf  # Import TensorFlow after Scipy or Scipy will break
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import imshow
from six.moves import cPickle as pickle
from PIL import Image
from datetime import datetime
#import glob
#from scipy import misc

from download import *

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image_RGB( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "RGB" )
    img.save( outfilename )

class CelebA_Dataset():
    def __init__(self, dict):
        self.train_images = dict['train']
        self.test_images = dict['test']
        self.validation_images = dict['validation']

def open_pickle(pickle_filepath):
    with open(pickle_filepath, 'rb') as f:
        result = pickle.load(f)
        celebA = CelebA_Dataset(result)
        del result
    return celebA

def crop_random(image_ori, width=90,height=90, x=None, y=None):
    if image_ori is None: return None
    thisheight = image_ori.shape[0]
    thiswidth = image_ori.shape[1]
    random_y = np.random.randint(0,thisheight-height) if x is None else x
    random_x = np.random.randint(0,thiswidth-width) if y is None else y

    image = image_ori.copy()
    crop = image_ori.copy()
    crop = crop[random_y:random_y+height, random_x:random_x+width]
    image[random_y:random_y+height, random_x:random_x+width, 0] = 2*117. / 255. - 1.
    image[random_y:random_y+height, random_x:random_x+width, 1] = 2*104. / 255. - 1.
    image[random_y:random_y+height, random_x:random_x+width, 2] = 2*123. / 255. - 1.

    return image, crop, random_x, random_y

def crop_random_batch(batch_samples, fixed):
    
    image_all = []
    cropped_all = []
    crop_path = "./cropped_images"
    if not os.path.exists(crop_path):
        os.mkdir(crop_path)
        
    batch_size = len(batch_samples)
    for i in range(batch_size):
        image_ori = load_image(batch_samples[i])
        image_name = os.path.splitext(batch_samples[i].split("/")[-1])[0]
        image_name = image_name + '.jpg'
        if fixed:
            image, crop, _, _ = crop_random(image_ori)
        else:
            width = np.random.randint(1,91)
            image, crop, _, _ = crop_random(image_ori, width=width, height=width)
        cropped_all.append(image)
        image_all.append(image_ori)
        crop_image_path = os.path.join(crop_path, image_name)
        #print(crop_image_path)
        save_image_RGB(image, crop_image_path)

    cropped_all = np.array(cropped_all) / 127.5 - 1
    image_all = np.array(image_all) / 127.5 - 1 
    return cropped_all, image_all

def save_results(results, batch_samples, valid_test):
    batch_size = results.shape[0]
    if valid_test == 1:
    	result_path = "./test_images"
    else:
    	result_path = "./valid_images"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for i in range(batch_size):
        image_name = os.path.splitext(batch_samples[i].split("/")[-1])[0]
        image_name = image_name + '.jpg'
        result_image_path = os.path.join(result_path, image_name)
        result = ((results[i] + 1) * 127.5).astype('uint8')
        save_image_RGB(result, result_image_path)

class Model():
    def __init__(self):
        self.num_epoch = 15
        self.batch_size = 64
        self.log_step = 500
        self.learning_rate = 1e-4
        self.real_input = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.real_label = tf.placeholder(tf.float32, [None, 128, 128, 3])        
        self.is_train = tf.placeholder(tf.bool)        
        self._init_ops()

    def new_conv_layer( self, bottom, filter_shape, activation=tf.identity, padding='SAME', stride=1, name=None ):
        with tf.variable_scope(name):
            w = tf.get_variable(
                    "W",
                    shape=filter_shape,
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=filter_shape[-1],
                    initializer=tf.constant_initializer(0.))

            conv = tf.nn.conv2d( bottom, w, [1,stride,stride,1], padding=padding)
            bias = activation(tf.nn.bias_add(conv, b))

        return bias #relu

    def new_deconv_layer(self, input, kernel_size, stride, num_filter, name):
        with tf.variable_scope(name):
            stride_shape = [1, stride, stride, 1]
            filter_shape = [kernel_size, kernel_size, num_filter, input.get_shape()[3]]
            output_shape = tf.stack([tf.shape(input)[0], tf.shape(input)[1] * 2, tf.shape(input)[2] * 2, num_filter])

            W = tf.get_variable('w', filter_shape, tf.float32, tf.random_normal_initializer(0.0, 0.02))
            b = tf.get_variable('b', [1, 1, 1, num_filter], initializer = tf.constant_initializer(0.0))
            
        return tf.nn.conv2d_transpose(input, W, output_shape, stride_shape, padding = 'SAME') + b
        
    def new_fc_layer( self, bottom, output_size, name ):
        shape = bottom.get_shape().as_list()
        dim = np.prod( shape[1:] )
        x = tf.reshape( bottom, [-1, dim])
        input_size = dim

        with tf.variable_scope(name):
            w = tf.get_variable(
                    "W",
                    shape=[input_size, output_size],
                    initializer=tf.random_normal_initializer(0., 0.005))
            b = tf.get_variable(
                    "b",
                    shape=[output_size],
                    initializer=tf.constant_initializer(0.))
            fc = tf.nn.bias_add(tf.matmul(x, w), b)

        return fc

    def channel_wise_fc_layer(self, input, name): # bottom: (4x4x512)
        _, width, height, n_feat_map = input.get_shape().as_list()
        input_reshape = tf.reshape( input, [-1, width*height, n_feat_map] )
        input_transpose = tf.transpose( input_reshape, [2,0,1] )

        with tf.variable_scope(name):
            W = tf.get_variable(
                    "W",
                    shape=[n_feat_map,width*height, width*height], # (512,16,16)
                    initializer=tf.random_normal_initializer(0., 0.005))
            #output = tf.batch_matmul(input_transpose, W)
            output = tf.matmul(input_transpose, W)

        output_transpose = tf.transpose(output, [1,2,0])
        output_reshape = tf.reshape( output_transpose, [-1, height, width, n_feat_map] )

        return output_reshape

    def leaky_relu(self, bottom, leak=0.1):
        return tf.maximum(leak*bottom, bottom)

    def batchnorm(self, input, is_training):
        out = tf.contrib.layers.batch_norm(input, decay = 0.99, center = True, scale = True,
                                           is_training = is_training, updates_collections = None)
        return out

    '''
    def batchnorm(self, bottom, is_train, epsilon=1e-8, name=None):
        bottom = tf.clip_by_value( bottom, -100., 100.)
        depth = bottom.get_shape().as_list()[-1]

        with tf.variable_scope(name):

            gamma = tf.get_variable("gamma", [depth], initializer=tf.constant_initializer(1.))
            beta  = tf.get_variable("beta" , [depth], initializer=tf.constant_initializer(0.))

            batch_mean, batch_var = tf.nn.moments(bottom, [0,1,2], name='moments')
            ema = tf.train.ExponentialMovingAverage(decay=0.5)


            def update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            ema_apply_op = ema.apply([batch_mean, batch_var])
            ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)
            mean, var = tf.cond(
                    is_train,
                    update,
                    lambda: (ema_mean, ema_var) )

            normed = tf.nn.batch_norm_with_global_normalization(bottom, mean, var, beta, gamma, epsilon, False)
        return normed
    '''

    def _model( self, images, is_train ):
        batch_size = images.get_shape().as_list()[0]

        with tf.variable_scope('model'):
            conv1 = self.new_conv_layer(images, [4,4,3,64], stride=2, name="conv1" ) 
            self.bn1 = self.leaky_relu(self.batchnorm(conv1, is_train))  # batch * 64 * 64 * 64
            print('conv1 layer: ' + str(self.bn1.get_shape()))
            conv2 = self.new_conv_layer(self.bn1, [4,4,64,64], stride=2, name="conv2" ) 
            self.bn2 = self.leaky_relu(self.batchnorm(conv2, is_train))  # batch * 32 * 32 * 64
            print('conv2 layer: ' + str(self.bn2.get_shape()))
            conv3 = self.new_conv_layer(self.bn2, [4,4,64,128], stride=2, name="conv3" ) 
            self.bn3 = self.leaky_relu(self.batchnorm(conv3, is_train))   # batch * 16 * 16 * 128
            print('conv3 layer: ' + str(self.bn3.get_shape()))
            conv4 = self.new_conv_layer(self.bn3, [4,4,128,256], stride=2, name="conv4")
            self.bn4 = self.leaky_relu(self.batchnorm(conv4, is_train))   # batch * 8 * 8 * 512
            print('conv4 layer: ' + str(self.bn4.get_shape()))
            conv5 = self.new_conv_layer(self.bn4, [4,4,256,512], stride=2, name="conv5")
            self.bn5 = self.leaky_relu(self.batchnorm(conv5, is_train))   # batch * 4 * 4 * 512
            print('conv5 layer: ' + str(self.bn5.get_shape()))
            
            self.cwfc = self.channel_wise_fc_layer(self.bn5, name='cwfc') # batch * 4 * 4 * 512
            print('channel wise fc layer: ' + str(self.cwfc.get_shape()))
            #conv5 = self.new_conv_layer(bn4, [4,4,,512], stride=2, name="conv5")
            #bn5 = self.leaky_relu(self.batchnorm(conv5, is_train, name='bn5'))
            #conv6 = self.new_conv_layer(bn5, [4,4,512,4000], stride=2, padding='VALID', name='conv6')
            #bn6 = self.leaky_relu(self.batchnorm(conv6, is_train, name='bn6'))

            print(type(self.cwfc))
            #deconv4 = self.new_deconv_layer( bn6, [4,4,256,512], cwfc.get_shape().as_list(), padding='VALID', stride=2, name="deconv4")
            #debn4 = tf.nn.relu(self.batchnorm(deconv4, is_train, name='debn4'))
            deconv4 = self.new_deconv_layer( self.cwfc, 4, 2, 256, name="deconv4")
            self.debn4 = tf.nn.relu(self.batchnorm(deconv4, is_train))
            print('deconv4 layer: ' + str(self.debn4.get_shape()))
            deconv3 = self.new_deconv_layer( self.debn4, 4, 2, 128, name="deconv3")
            self.debn3 = tf.nn.relu(self.batchnorm(deconv3, is_train))
            print('deconv3 layer: ' + str(self.debn3.get_shape()))
            deconv2 = self.new_deconv_layer( self.debn3, 4, 2, 64, name="deconv2")
            self.debn2 = tf.nn.relu(self.batchnorm(deconv2, is_train))
            print('deconv2 layer: ' + str(self.debn2.get_shape()))
            deconv1 = self.new_deconv_layer( self.debn2, 4, 2, 64, name="deconv1")
            self.debn1 = tf.nn.relu(self.batchnorm(deconv1, is_train))
            print('deconv1 layer: ' + str(self.debn1.get_shape()))
            self.recon = self.new_deconv_layer( self.debn1, 4, 2, 3, name="recon")
            print('recon layer: ' + str(self.recon.get_shape()))

        return tf.nn.tanh(self.recon)

    # Define operations
    def _init_ops(self):
        
        self.result = self._model(self.real_input, self.is_train)
        self.loss_op = tf.reduce_sum(tf.square(self.real_label - self.result))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.minimize(self.loss_op)
        
    def train(self, sess, train_samples, valid_samples):
        sess.run(tf.global_variables_initializer())
        num_train = len(train_samples)
        step = 0
        
        for epoch in range(self.num_epoch):
            loss_train = 0
            #results = None
            for i in range(num_train // self.batch_size):
                step += 1
                
                batch_samples, batch_labels = crop_random_batch(train_samples[i * self.batch_size : (i + 1) * self.batch_size], fixed=True)
                #batch_labels = train_labels[i * self.batch_size : (i + 1) * self.batch_size]
                feed_dict = {
                    self.real_input: batch_samples,
                    self.real_label: batch_labels,
                    self.is_train: True
                }
                
                fetches = [self.train_op, self.loss_op]
                _, loss = sess.run(fetches, feed_dict)
                loss = loss / self.batch_size
                #plot_s = plot_s * smooth_factor + loss * (1 - smooth_factor)
                #plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)
                #losses_plot.append(loss / plot_ws)
                
                loss_train += loss
                
                if step % self.log_step == 0:
                    print('Iteration {0}: loss = {1:.4f}'.format(step, loss))
                #loss_train += loss
            '''
            #print(train_samples.shape)
            '''
            loss_train /= (num_train // self.batch_size)
            #losses_total.append(loss_train)
            '''
            plt.plot(losses_plot)
            plt.title('generation loss')
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.show()'''
            
            print("epch {0} has finished!".format(epoch))
            self.test(sess, valid_samples, 0)
            #print(results.shape)
            #losses_plot = []
            
        
    def test(self, sess, test_samples, valid_test):
        loss_test = 0
        num_test = len(test_samples)
        results = None
        for i in range(num_test // self.batch_size):
            batch_samples, batch_labels = crop_random_batch(test_samples[i * self.batch_size : (i + 1) * self.batch_size], fixed=False)
            #batch_labels = test_labels[i * self.batch_size : (i + 1) * self.batch_size]
            feed_dict = {
                self.real_input: batch_samples,
                self.real_label: batch_labels,
                self.is_train: False
            }
            fetches = [self.loss_op, self.result]
            loss, result = sess.run(fetches, feed_dict) 
            loss = loss / self.batch_size           
            #print(result[0])
            save_results(result, test_samples[i * self.batch_size : (i + 1) * self.batch_size], valid_test)
            if i == 0: 
                results = result
            else:
                results = np.concatenate((results, result))
            loss_test += loss
        loss_test /= (num_test // self.batch_size)
        print('the loss is {0}'.format(loss_test))
        return results


if __name__ == '__main__':

    base_path = '.'
    #prepare_data_dir()
    #download_celeb_a(base_path)
    #resize_images(base_path, 128)
    #Pickle_Dataset(base_path)


    start_time = datetime.now()
    pickle_filepath = "./celebA.pickle"
    celebA = open_pickle(pickle_filepath)
    	#:tf.reset_default_graph()
	
    with tf.Session() as sess:
        model = Model()
        sess.run(tf.global_variables_initializer())
        model.train(sess, celebA.train_images, celebA.validation_images)
        results = model.test(sess, celebA.test_images, 1)
        saver = tf.train.Saver()
        saver.save(sess, 'model/imgCNN_sample.ckpt')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))



