# Modified from https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

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

def crop_random(image_ori, width=32,height=32, x=None, y=None):
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

'''
def crop_random_batch(batch_samples):
    
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
        image, crop, _, _ = crop_random(image_ori)
        cropped_all.append(image)
        image_all.append(image_ori)
        crop_image_path = os.path.join(crop_path, image_name)
        #print(crop_image_path)
        save_image_RGB(image, crop_image_path)

    cropped_all = np.array(cropped_all) / 127.5 - 1
    image_all = np.array(image_all) / 127.5 - 1 
    return cropped_all, image_all

'''

def crop_random_batch(batch_samples, sub_batch_resize):
    
    image_all = []
    cropped_all = []
    crop_path = "./cropped_images_2tr"
    if not os.path.exists(crop_path):
        os.mkdir(crop_path)
        
    #batch_size = len(batch_samples)
    difficulty_level = sub_batch_resize.shape[0]
    #sub_batch_size = np.copy(sub_batch_resize)
    unit = 18
    numb = 0

    for i in range(difficulty_level):
        for j in range(sub_batch_resize[i]):
            width = np.random.randint(i * unit+1,(i+1) * unit+1)
            height = width
    #for i in range(batch_size):
            image_ori = load_image(batch_samples[numb])
            image_name = os.path.splitext(batch_samples[numb].split("/")[-1])[0]
            image_name = image_name + '_' + str(width) + '.jpg'

        #level = i % (batch_size / difficulty_level)

            image, crop, _, _ = crop_random(image_ori, width, height)
            cropped_all.append(image)
            image_all.append(image_ori)
            crop_image_path = os.path.join(crop_path, image_name)
        #print(crop_image_path)
            save_image_RGB(image, crop_image_path)
            numb += 1

    cropped_all = np.array(cropped_all) / 127.5 - 1
    image_all = np.array(image_all) / 127.5 - 1 
    return cropped_all, image_all

def crop_random_batch_mix(batch_samples):
    image_all = []
    cropped_all = []
    crop_path = "./cropped_images_2te"
    if not os.path.exists(crop_path):
        os.mkdir(crop_path)
    
    width_whole = 90 # 18 * 5
    batch_size = len(batch_samples)
    for i in range(batch_size):
        image_ori = load_image(batch_samples[i])
        image_name = os.path.splitext(batch_samples[i].split("/")[-1])[0]
        width = np.random.randint(1, width_whole+1)
        image_name = image_name + '_' + str(width) + '.jpg'
        
        image, crop, _, _ = crop_random(image_ori, width, width)
        cropped_all.append(image)
        image_all.append(image_ori)
        crop_image_path = os.path.join(crop_path, image_name)
        #print(crop_image_path)
        save_image_RGB(image, crop_image_path)

    cropped_all = np.array(cropped_all) / 127.5 - 1
    image_all = np.array(image_all) / 127.5 - 1 
    return cropped_all, image_all

def PSNR(image1, image2, difficulty_level, sub_batch_size):
    #z = np.product(image1.shape)
    PSNR = []
    num = 0
    MSE = np.mean(np.power((image1 - image2),2), axis = (1,2,3))
    for i in range(difficulty_level):
        MSE_sub = MSE[num:num + sub_batch_size[i]]
        PIXEL_MAX = 255.0
        PSNR.append(np.mean(20 * np.log10(PIXEL_MAX/np.sqrt(MSE_sub))))
        num += sub_batch_size[i]
    reciprocal_PSNR = 1 / (np.array(PSNR))
    batch_size = np.sum(sub_batch_size)
    Sub_batch_size = (batch_size * reciprocal_PSNR / (np.sum(reciprocal_PSNR))).astype('uint8')
    Sub_batch_size[Sub_batch_size.shape[0] - 1] += batch_size - np.sum(Sub_batch_size)
    return Sub_batch_size

def save_results(results, batch_samples, valid_test):
    batch_size = results.shape[0]
    if valid_test == 1:
    	result_path = "./test_images_MW"
    elif valid_test == 0:
    	result_path = "./valid_images_MW"
    else:
        result_path = "./train_images_MW"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for i in range(batch_size):
        image_name = os.path.splitext(batch_samples[i].split("/")[-1])[0]
        image_name = image_name + '.jpg'
        result_image_path = os.path.join(result_path, image_name)
        result = ((results[i] + 1) * 127.5).astype('uint8')
        save_image_RGB(result, result_image_path)
    #print("saved")

'''
class GANModel():
    def __init__(self):
        self.num_epoch = 1
        self.batch_size = 100
        self.log_step = 50
        self.learning_rate = 1e-4
        self.real_input = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.real_label = tf.placeholder(tf.float32, [None, 128, 128, 3])        
        self.is_train = tf.placeholder(tf.bool)        
        self._init_ops()

    def conv(self, batch_input, out_channels, stride):
        with tf.variable_scope("conv"):
            in_channels = batch_input.get_shape()[3]
            filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
            # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
            #     => [batch, out_height, out_width, out_channels]
            padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
            conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
            return conv

    def lrelu(self, x, a):
        with tf.name_scope("lrelu"):
            # adding these together creates the leak part and linear part
            # then cancels them out by subtracting/adding an absolute value term
            # leak: a*x/2 - a*abs(x)/2
            # linear: x/2 + abs(x)/2

            # this block looks like it has 2 inputs on the graph unless we do this
            x = tf.identity(x)
            return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


    def batchnorm(self, input):
        with tf.variable_scope("batchnorm"):
            # this block looks like it has 3 inputs on the graph unless we do this
            input = tf.identity(input)

            channels = input.get_shape()[3]
            offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
            scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
            mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
            variance_epsilon = 1e-5
            normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
            return normalized

    def deconv(self, batch_input, out_channels):
        with tf.variable_scope("deconv"):
            batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
            filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
            # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
            #     => [batch, out_height, out_width, out_channels]
            conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
            return conv

    def generator(self, input):
        with tf.variance_scope("encoder1"):
'''



class GANModel():
    def __init__(self):
        self.num_epoch = 15
        self.batch_size = 90
        self.log_step = 500
        self.learning_rate = tf.placeholder(tf.float32)
        self.real_input = tf.placeholder(tf.float32, [None, 128, 128, 3])
        self.real_label = tf.placeholder(tf.float32, [None, 128, 128, 3])        
        self.is_train = tf.placeholder(tf.bool) 
        self.gan_weight = 1
        self.l1_weight = 100 
        self.l1_weight_crop = 500
        self._dis_called = False
        self._gen_called = False
        self.ntf = 64
        self.difficulty_level = 5
        self.sub_batch_size = np.array([self.batch_size // self.difficulty_level] * self.difficulty_level)
        self._init_ops()
        

    def new_conv_layer( self, bottom, kernel_size, output_channels, activation=tf.identity, padding='SAME', stride=1, name=None ):
        filter_shape = [kernel_size, kernel_size, bottom.shape[3], output_channels]
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
    '''    
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
    '''
    
    def new_fc_layer(self, input, num_output, name):
        with tf.variable_scope(name):
            num_input = input.get_shape()[1]
            W = tf.get_variable('w', [num_input, num_output], tf.float32, tf.random_normal_initializer(0.0, 0.02))
            b = tf.get_variable('b', [num_output], initializer = tf.constant_initializer(0.0))
            return tf.matmul(input, W) + b

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

    def leaky_relu(self, bottom, leak=0.2):
        return tf.maximum(leak*bottom, bottom)

    def batch_norm(self, input, is_training):
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

    '''
        modified from pix2pix: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
    '''

    def build_generator( self, images, is_train ):
        #batch_size = images.get_shape().as_list()[0]
        layers = []
        i = 1;
        with tf.variable_scope("encoder_1"):
            output = self.new_conv_layer(images, 4, self.ntf, stride = 2, name = "conv")
            print('conv' + str(i) + 'layer: ' + str(output.get_shape()))
            layers.append(output)

        layers_specs = [
            self.ntf * 2,       # encoder_2: batch * 64 * 64 * 128  
            self.ntf * 4,       # encoder_3: batch * 32 * 32 * 256 
            self.ntf * 8,       # encoder_4: batch * 16 * 16 * 512
            self.ntf * 8,       # encoder_5: batch * 8 * 8 * 512
            self.ntf * 8,       # encoder_6: batch * 4 * 4 * 512
            self.ntf * 8,       # encoder_7: batch * 2 * 2 * 512
        ]

        for output_channels in layers_specs:
            with tf.variable_scope("encoder_" + str(len(layers)+1)):
                relu = self.leaky_relu(layers[-1])
                conv = self.new_conv_layer(relu, 4, output_channels, stride = 2, name = "conv")
                output = self.batch_norm(conv, is_train)
                print('conv' + str(len(layers)+1) + 'layer: ' + str(output.get_shape()))
                layers.append(output)

        layers_specs = [
            (self.ntf * 8, 0.5),    #decoder_7: batch * 1 * 1 * 512 => batch * 2 * 2 * (512 * 2)
            (self.ntf * 8, 0.5),    #decoder_6: batch * 2 * 2 * (512 * 2) => batch * 4 * 4 * (512 * 2)
            (self.ntf * 8, 0.0),    #decoder_5: batch * 4 * 4 * (512 * 2) => batch * 8 * 8 * (512 * 2)
            (self.ntf * 4, 0.0),    #decoder_4: batch * 8 * 8 * (512 * 2) => batch * 16 * 16 * (256 * 2)
            (self.ntf * 2, 0.0),    #decoder_3: batch * 16 * 16 * (256 * 2) => batch * 32 * 32 * (128 * 2)
            (self.ntf, 0.0),        #decoder_2: batch * 32 * 32 * (128 * 2) => batch * 64 * 64 * (64 * 2)
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (output_channels, dropout) in enumerate(layers_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_" + str(skip_layer + 1)):
                
                if decoder_layer == 0:
                    # first decode layer doesn't skip connections
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], 0.5 * layers[skip_layer]], axis = 3)

                #input = layers[-1]

                relu = tf.nn.relu(input)
                output = self.new_deconv_layer(relu, 4, 2, output_channels, name = "deconv")
                output = self.batch_norm(output, is_train)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob = 1 - dropout)

                print('deconv' + str(skip_layer+1) + 'layer: ' + str(output.get_shape()))
                layers.append(output)

        # decode_1: batch * 64 * 64 * (64 * 2) => batch * 128 * 128 * 3
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis = 3)
            relu = tf.nn.relu(input)
            output = self.new_deconv_layer(relu, 4, 2, 3, name = "deconv")
            output = tf.tanh(output)
            print('deconv1 layer: ' + str(output.get_shape()))
            layers.append(output)

        return layers[-1]

    def build_discriminator(self, images, is_train):
        #with tf.variable_scope('DIS', reuse = self._dis_called):
        n_layers = 2
        layers = []

        # 2 batch * 128 * 128 * 3 => batch * 128 * 128 * (3 * 2)
        #input = tf.concat([images, labels], axis = 3)

        # layer_1: batch * 128 * 128 * (3*2) => batch * 64 * 64 * 64
        with tf.variable_scope("dis_layer_1"):
            conv = self.new_conv_layer(images, 4, self.ntf, stride = 2, name = "conv")
            relu = self.leaky_relu(conv)
            print('dis_conv1 layer: ' + str(relu.get_shape()))
            layers.append(relu)

        # layer_2: batch * 64 * 64 * 64 -> batch * 32 * 32 * 128
        # layer_3: batch * 32 * 32 * 128 -> batch * 31 * 31 * 256
        for i in range(n_layers):
            with tf.variable_scope("dis_layer_" + str(i+2)):
                output_channels = self.ntf * (2 ** (i+1))
                stride = 1 if i == n_layers -1 else 2
                conv = self.new_conv_layer(layers[-1], 4, output_channels, stride = stride, name = "conv")
                output = self.leaky_relu(self.batch_norm(conv, is_train))
                print('dis_conv' + str(i+2) + ' layer: ' + str(output.get_shape()))
                layers.append(output)

        # layer_4: batch * 31 * 31 * 256 -> batch * 30 * 30 * 1
        with tf.variable_scope("dis_layer_4"):
            conv = self.new_conv_layer(layers[-1], 4, 1, stride = 1, name = "conv")
            output = tf.sigmoid(conv)
            print('dis_conv4 layer: ' + str(output.get_shape()))
            layers.append(output)

        return layers[-1]

    # Define operations
    def _init_ops(self):
        # generator
        with tf.variable_scope("GEN"):
            self.result = self.build_generator(self.real_input, self.is_train)

        #real discriminator
        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("DIS"):
                real_dis = self.build_discriminator(self.real_label, self.is_train)
        
        #self._dis_called = True
        #false discriminator
        #with tf.name_scope("false_discriminator"):
        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("DIS", reuse=True):
                false_dis = self.build_discriminator(self.result, self.is_train)

        # discriminator loss
        #with tf.name_scope("discriminator_loss"):
        self.dis_loss_op = tf.reduce_mean(-(tf.log(real_dis + 1e-12) + tf.log(1 - false_dis + 1e-12)))
        
        #with tf.name_scope("generator_loss"):
        gen_loss_GAN = tf.reduce_mean(-tf.log(false_dis + 1e-12))
        gen_all_diff_L1 = tf.abs(self.real_label - self.result)
        gen_all_loss_L1 = tf.reduce_mean(gen_all_diff_L1)

        '''
        if gen_all_loss_L1 < 0.13:
            gen_crop_diff_L1 = tf.relu(gen_all_diff_L1 - 0.1)
            gen_crop_diff_L1 = tf.reduce_mean(gen_crop_diff_L1)
            self.l1_weight_crop = 500
        else:
            self.l1_weight_crop = 0
        '''
        #y = tf.constant(0.13)
        self.l1_weight_crop = 13. / gen_all_loss_L1

        gen_crop_diff_L1 = tf.nn.relu(gen_all_diff_L1 - 0.1)
        gen_crop_diff_L1 = tf.reduce_mean(gen_crop_diff_L1)
        
        self.gen_loss_op = gen_loss_GAN * self.gan_weight + gen_all_loss_L1 * self.l1_weight + gen_crop_diff_L1 * self.l1_weight_crop

        #with tf.name_scope("discriminator_train"):
        dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'DIS')
        dis_optim = tf.train.AdamOptimizer(self.learning_rate)
        dis_grads_and_vars = dis_optim.compute_gradients(self.dis_loss_op, var_list=dis_vars)
        self.dis_train_op = dis_optim.apply_gradients(dis_grads_and_vars)

        #with tf.name_scope("generator_train"):
        with tf.control_dependencies([self.dis_train_op]):
            gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'GEN')
            gen_optim = tf.train.AdamOptimizer(self.learning_rate)
            gen_grads_and_vars = gen_optim.compute_gradients(self.gen_loss_op, var_list=gen_vars)
            self.gen_train_op = gen_optim.apply_gradients(gen_grads_and_vars)

    def train(self, sess, train_samples, valid_samples):
        sess.run(tf.global_variables_initializer())
        num_train = len(train_samples)
        step = 0

        learning_rate = 1e-4

        start1_time = datetime.now()
        
        #smooth_factor = 0.95
        #plot_dis_s = 0
        #plot_gen_s = 0
        #plot_ws = 0
        
        #dis_losses = []
        #gen_losses = []
        for epoch in range(self.num_epoch):
            dis_loss_train = 0
            gen_loss_train = 0

            
            #results = None
            for i in range(num_train // self.batch_size):
                
                
                batch_samples, batch_labels = crop_random_batch(train_samples[i * self.batch_size : (i + 1) * self.batch_size], self.sub_batch_size)
                #batch_labels = train_labels[i * self.batch_size : (i + 1) * self.batch_size]
                dis_feed_dict = {
                    self.learning_rate: learning_rate,
                    self.real_input: batch_samples,
                    self.real_label: batch_labels,
                    self.is_train: True
                }

                _, dis_loss = sess.run([self.dis_train_op, self.dis_loss_op], feed_dict = dis_feed_dict)
                

                gen_feed_dict = {
                    self.learning_rate: learning_rate,
                    self.real_input: batch_samples,
                    self.real_label: batch_labels,
                    self.is_train: True,
                }

                _, gen_loss, result = sess.run([self.gen_train_op, self.gen_loss_op, self.result], feed_dict = gen_feed_dict)
                
                results = (result + 1) * 127.5
                batch_labels = (batch_labels + 1) * 127.5
                self.sub_batch_size = PSNR(batch_labels,results, self.difficulty_level, self.sub_batch_size)
                #lot_dis_s = plot_dis_s * smooth_factor + dis_loss * (1 - smooth_factor)
                #plot_gen_s = plot_gen_s * smooth_factor + gen_loss * (1 - smooth_factor)
                #plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)
                #dis_losses.append(plot_dis_s / plot_ws)
                #gen_losses.append(plot_gen_s / plot_ws)
                
                #plot_s = plot_s * smooth_factor + loss * (1 - smooth_factor)
                #plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)
                #losses_plot.append(loss / plot_ws)
                #dis_loss = self.batch_size
                #gen_loss = self.batch_size
                #dis_loss /= self.batch_size
                #gen_loss /= self.batch_size
                dis_loss_train += dis_loss
                gen_loss_train += gen_loss

                
                #print('Duration: {}'.format(end1_time - start1_time))
                if step % self.log_step == 0:
                    end1_time = datetime.now()
                    f = open("log_result.txt", "a+")
                    f.write('Iteration {0}: dis_loss = {1:.4f}, gen_loss = {2:.4f}\n'.format(step, dis_loss, gen_loss))
                    f.write('Duration: {}'.format(end1_time - start1_time))
                    print('Iteration {0}: dis_loss = {1:.4f}, gen_loss = {2:.4f}'.format(step, dis_loss, gen_loss))
                    #print(self.sub_batch_size)
                    save_results(result, train_samples[i * self.batch_size : (i + 1) * self.batch_size], 2)
                    f.close()
                    start1_time = datetime.now()
                step += 1
                #loss_train += loss
            '''
            #print(train_samples.shape)
            '''
            
            dis_loss_train /= (num_train // self.batch_size)
            gen_loss_train /= (num_train // self.batch_size)
            
            #fig = plt.figure(figsize = (8, 8))   
            #ax1 = plt.subplot(111)
            #ax1.imshow(viz_grid(self.generate(self.tracked_noise), 1))
            #plt.show()

            #plt.plot(dis_losses)
            #plt.title('discriminator loss')
            #plt.xlabel('iterations')
            #plt.ylabel('loss')
            #plt.show()

            #plt.plot(gen_losses)
            #plt.title('generator loss')
            #plt.xlabel('iterations')
            #plt.ylabel('loss')
            #plt.show()
            
            #losses_total.append(loss_train)
            '''
            plt.plot(losses_plot)
            plt.title('generation loss')
            plt.xlabel('iterations')
            plt.ylabel('loss')
            plt.show()'''
            
            print("epoch {0} has finished! dis_loss = {1:.4f}, gen_loss = {2:.4f}".format(epoch, dis_loss_train, gen_loss_train))
            
            self.test(sess, valid_samples, 0)
            #print(results.shape)
            #losses_plot = []
            learning_rate *= 0.97

            
        
    def test(self, sess, test_samples, valid_test):
        num_test = len(test_samples)
        #results = None
        l2_loss_test = 0
        PSNR_loss_test = 0
        for i in range(num_test // self.batch_size):
            batch_samples, batch_labels = crop_random_batch_mix(test_samples[i * self.batch_size : (i + 1) * self.batch_size])
            #batch_labels = test_labels[i * self.batch_size : (i + 1) * self.batch_size]
            '''dis_feed_dict = {
                self.real_input: batch_samples,
                self.real_label: batch_labels,
                self.is_train: False
            }

            _, dis_loss = sess.run([self.dis_train_op, self.dis_loss_op], feed_dict = dis_feed_dict)
            
            '''


            gen_feed_dict = {
                self.real_input: batch_samples,
                self.real_label: batch_labels,
                self.is_train: False,
            }

            result = sess.run(self.result, feed_dict = gen_feed_dict)

            save_results(result, test_samples[i * self.batch_size : (i + 1) * self.batch_size], valid_test)

            result = (result + 1) * 127.5
            batch_labels = (batch_labels + 1) * 127.5

            l2_loss = np.mean(np.power((result - batch_labels),2), axis = (1,2,3))

            PSNR_loss = 20 * np.log10(255.0/np.sqrt(l2_loss))

            l2_loss_test += np.sum(l2_loss)

            PSNR_loss_test += np.sum(PSNR_loss)

            #plot_s = plot_s * smooth_factor + loss * (1 - smooth_factor)
            #plot_ws = plot_ws * smooth_factor + (1 - smooth_factor)
            #losses_plot.append(loss / plot_ws)
            #dis_loss /= self.batch_size
            #gen_loss /= self.batch_size
            #dis_loss_test += dis_loss
            #gen_loss_test += gen_loss

            #if i == 0: 
            #    results = result
            #else:
            #    results = np.concatenate((results, result))

        l2_loss_test /= (num_test // self.batch_size) * self.batch_size
        PSNR_loss_test /= (num_test // self.batch_size) * self.batch_size
        f = open("log_result.txt", "a+")
        f.write("Evaluation: \nthe average l2 loss is {0} the average PSNR is {1}".format(l2_loss_test, PSNR_loss_test))
        f.close()
        print('Evaluation time:')
        print('the average l2 loss is {0}'.format(l2_loss_test))
        print('the average PSNR is {0}'.format(PSNR_loss_test))


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

        model = GANModel()
        sess.run(tf.global_variables_initializer())
        model.train(sess, celebA.train_images, celebA.validation_images)
        model.test(sess, celebA.test_images, 1)
        saver = tf.train.Saver()
        saver.save(sess, 'model/GAN_OD_sample.ckpt')
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))



