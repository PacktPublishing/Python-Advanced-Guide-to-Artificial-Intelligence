'''
Created on 16-May-2018

@author: aii32199
'''
'''
Created on 11-May-2018

@author: aii32199
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

def montage(images):
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    m = np.ones((images.shape[1] * n_plots + n_plots + 1, images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m


tf.reset_default_graph()

keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
is_training = tf.placeholder(dtype=tf.bool, name='is_training')

#Following function is for leaky-relu
def LeakyRelu(x,alpha=0.2):
    
    #X: is input tensor
    #ALPHA: is leaky value
    
    #We need to just multiply the alpha value to input tensor    
    value = tf.multiply(x, alpha)
    
    #Return the maximum of two values 
    return tf.maximum(x, value)

def binary_cross_entropy(x, z):
    #X: Expected output of network
    #Z: Actual output of the network
    
    #Define a small value epsilon prevent divide by zero error
    eps = 1e-12
    
    #Calculate the loss
    return (-(x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))

#Following function will create a Discriminator
def Discriminator(img_in, reuse=None, prob=keep_prob,ksize = 5):
    
    #IMG_IN: is input image tensor.
    #REUSE: is for reusing same variables, as we need to create-
    #two copies of discriminator network it will necessary to define.
    #PROB: is drop-out probability.
    #KSIZE: is convolution kernel size we will choose 5X5 this time.
    
    #We will start with defining activation function for our model
    activation = LeakyRelu
    
    #We will set the variable scope for the discriminator network. 
    with tf.variable_scope("discriminator", reuse=reuse):
        
        #We will start reshaping our input data into 1X28X28 from 28X28X1
        x = tf.reshape(img_in, shape=[-1, 28, 28, 1])
        
        #Here we will add first Convolution + ReLu with 64 filters,
        #kernel size will be 5X5 and a stride of 2,
        #it will reduce the size of image by half of its original size.
        #we will use 'same' padding which will retain the size after convolution    
        x = tf.layers.conv2d(x, 
                             kernel_size=ksize, 
                             filters=64, strides=2, 
                             padding='same', 
                             activation=activation)
        
        #Now add some drop-out to the next layer
        #It will randomly remove some neurons from the training in each epoch 
        x = tf.layers.dropout(x, prob)#14X14X64
        
        #It will be our second Conv+ReLu again with the same configuaration as first
        #here we will not use strides so size of image will not reduce
        x = tf.layers.conv2d(x, 
                             kernel_size=ksize, 
                             filters=64, strides=1, 
                             padding='same', activation=activation)
        
        #Again add some drop-outs
        x = tf.layers.dropout(x, prob)#14X14X64
        
        #This is our third Conv+ReLu with same configuration as previous layer
        x = tf.layers.conv2d(x, 
                             kernel_size=ksize, 
                             filters=64, strides=1, 
                             padding='same', activation=activation)
        
        #Some more drop-outs
        x = tf.layers.dropout(x, prob)#14X14X64
        
        #Now we will convert 2D data into 1D tensor.
        x = tf.contrib.layers.flatten(x)
        
        #Here we will implement first dens layer with 128 neurons
        x = tf.layers.dense(x, 
                            units=128, activation=activation)#12544    
        
        #It will be our second dens layer with 1 neuron as it is a binary classifier
        x = tf.layers.dense(x, 
                            units=1, activation=tf.nn.sigmoid)#128            
        
        #Return the model
        return x#1    

def Generator(z, prob=keep_prob, is_training=is_training,ksize=5):
    
    #Z: is the random input vector.
    #PROB: is droput probability
    #IS_TRAINING: flag is for batch norm layer 
    #KSIZE: is convolution kernel size
    
    #We will use leaky relu activation
    activation = LeakyRelu
    
    #Add a regularization parameter for batch norm layer
    momentum = 0.99
    
    #Define the variable scope for generator
    with tf.variable_scope("generator", reuse=None):        
        x = z
        
        #we will define neurons for the dens layer,
        #there are two value d1 and d2 it will be used,
        #for reshaping flatten output of dense layer into 2D matrix
        d1 = 4
        d2 = 1
        
        #Define our first dense layer 
        x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
        
        #Add a dropout with defined probability
        x = tf.layers.dropout(x, prob)      
        
        #Here we add our batch normalization layer with regularization
        x = tf.contrib.layers.batch_norm(x, is_training=is_training,
                                          decay=momentum)
        
        #Now we will reshape our flatten tensor into 2D tensor  
        x = tf.reshape(x, shape=[-1, d1, d1, d2])
        
        #Resize the matrix with height and width = 7 
        x = tf.image.resize_images(x, size=[7, 7])
        
        #Now we will perform transposed convolution here with 64 filters of 5X5
        #Stride will be 2 with 'same' padding so upsampling will be done here 
        #tensor of 7X7 will be resized to 64X14X14
        x = tf.layers.conv2d_transpose(x, kernel_size=ksize, filters=64, 
                                       strides=2, padding='same', 
                                       activation=activation)
        
        #Add some dropout again
        x = tf.layers.dropout(x, prob)
        
        #Here will be our second batch norm layer
        x = tf.contrib.layers.batch_norm(x, is_training=is_training,
                                          decay=momentum)
        
        #Add another convolution layer with exact same hyper parameter as previous
        #So one more upsampling will happen here output will have size of 64X28X28
        x = tf.layers.conv2d_transpose(x, kernel_size=ksize, filters=64, 
                                       strides=2, padding='same', 
                                       activation=activation)
        
        #Add another dropout layer
        x = tf.layers.dropout(x, prob)
        
        #Here will be our third batch norm layer
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, 
                                         decay=momentum)
        
        #This will be our third transposed convolution layer,
        #this time stride size=1 so no upsampling will happen here 
        #output size of the tensor will be 64X28X28
        x = tf.layers.conv2d_transpose(x, kernel_size=ksize, filters=64, 
                                       strides=1, padding='same', 
                                       activation=activation)
        
        #Our final dropout layer
        x = tf.layers.dropout(x, prob)
        
        #Final batch norm layer
        x = tf.contrib.layers.batch_norm(x, is_training=is_training, 
                                         decay=momentum)
        
        #Final convolution with single filter of 5X5 
        #stride size =1 so no upsampling, output tensor size will be 1X28X28
        #Sigmoid activation will be used here 
        x = tf.layers.conv2d_transpose(x, kernel_size=ksize, 
                                       filters=1, strides=1,
                                       padding='same', 
                                       activation=tf.nn.sigmoid)
        
        #Return the Generator model
        return x
    


from tensorflow.contrib.layers import apply_regularization
from tensorflow.contrib.layers import*

class GAN():
    def __init__(self):
        
        #Lets start with defining the batch size.
        #it will same for noise and real data
        self.batch_size = 196
        self.n_noise = 196
        
        #We need to create 2 place holders to hold the data values for noise and data
        self.X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
        self.noise = tf.placeholder(dtype=tf.float32, shape=[None, self.n_noise])      
        
        #Here we call our generator to generate false data
        #we need to define dropout probability and training mode.
        self.g = Generator(self.noise, keep_prob, is_training)
        
        #Here we will create two dicriminator models.
        #these two will share same parameters(weights and biases).
        #one will operate on real data while other is on fake one.
        self.d_real = Discriminator(self.X_in)
        self.d_fake = Discriminator(self.g, reuse=True)
        
        #Separate the trainable variables for both generator as well as discriminator
        self.vars_g = [var for var in tf.trainable_variables() 
                       if var.name.startswith("generator")]
        self.vars_d = [var for var in tf.trainable_variables() 
                       if var.name.startswith("discriminator")]
        
        #Here we will apply some regularization on weights of 
        #generator and discriminator
        self.d_reg = apply_regularization(l2_regularizer(1e-6),self.vars_d)
        self.g_reg = apply_regularization(l2_regularizer(1e-6), self.vars_g)
        
        #We will use binary cross entropy loss to measure the performance
        #of our discriminators
        self.loss_d_real = binary_cross_entropy(tf.ones_like(self.d_real), self.d_real)
        self.loss_d_fake = binary_cross_entropy(tf.zeros_like(self.d_fake), self.d_fake)
        
        #Here we will calculate the loss for both networks
        self.loss_g = tf.reduce_mean(binary_cross_entropy(tf.ones_like(self.d_fake), self.d_fake))
        self.loss_d = tf.reduce_mean(0.5 * (self.loss_d_real + self.loss_d_fake))
        
        #Let's update the graphs
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        #Now its time to train the networks
        with tf.control_dependencies(self.update_ops):
            #Total loss of Discriminator with regularization            
            total_loss_d = self.loss_d + self.d_reg
            
            #Total loss of Generator with regularization
            total_loss_g = self.loss_g + self.g_reg
            
            #Set the learning rate
            lr = 0.00015

            #We will use RMSprop with SGD for training the networks                        
            self.optimizer_d = tf.train.RMSPropOptimizer(learning_rate=
                                                         lr).minimize(total_loss_d,
                                                                      var_list=self.vars_d)
            self.optimizer_g = tf.train.RMSPropOptimizer(learning_rate=
                                                         lr).minimize(total_loss_g,
                                                                      var_list=self.vars_g)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')        
def train(model):
    
    #We will start with creating a tensorflow session
    sess = tf.Session()
    
    #Initialize all the variables
    sess.run(tf.global_variables_initializer())
    
    #And hit the training loop
    for i in range(60000):
        
        #Set training flag on for batch normalization
        train_d = True
        train_g = True
        
        #Set Drop out probability
        keep_prob_train = 0.6 # 0.5
        
        #Here we will create random data to input generator
        n = np.random.uniform(0.0, 1.0, 
                              [model.batch_size,
                               model.n_noise]).astype(np.float32)
        
        #And here we will fetch input data from tensorflow 
        batch = [np.reshape(b, [28, 28]) 
                 for b in mnist.train.next_batch(batch_size=
                                                 model.batch_size)[0]]  
        
        #Lets start training by executing the session
        d_real_ls,d_fake_ls,g_ls,d_ls = sess.run([model.loss_d_real, 
                                                     model.loss_d_fake, 
                                                     model.loss_g, 
                                                     model.loss_d], 
                                                    feed_dict={model.X_in: 
                                                               batch, 
                                                               model.noise:
                                                               n, 
                                                               keep_prob: 
                                                               keep_prob_train, 
                                                               is_training:True})
        #Each epoch will give us loss by discriminator for real
        #and fake data take mean of it 
        d_real_ls = np.mean(d_real_ls)
        d_fake_ls = np.mean(d_fake_ls)
        g_ls = g_ls
        d_ls = d_ls
        
        #Here we will set some conditions so our discriminator
        #can not be strong than generator or vice versa
        #We will use the flags to stop training the discriminator or
        #generator 
        if g_ls * 1.5 < d_ls:
            train_g = False
            pass
        if d_ls * 2 < g_ls:
            train_d = False
            pass
        
        #Here we will limit the training
        if train_d:
            sess.run(model.optimizer_d, 
                     feed_dict={model.noise: n, 
                                model.X_in: batch, 
                                keep_prob: keep_prob_train, 
                                is_training:True})
            
            
        if train_g:
            sess.run(model.optimizer_g,
                      feed_dict={model.noise: n,
                                keep_prob: keep_prob_train, 
                                is_training:True})
            
        #Lets print out the results here
        #we will print all the losses after every 50 epochs  
        if not i % 50:
            print (i, d_ls, g_ls, d_real_ls, d_fake_ls)
            if not train_g:
                print("not training generator")
            if not train_d:
                print("not training discriminator")
            
            #Let's test the trained generator after every 50 iterations
            gen_img = sess.run(model.g, 
                               feed_dict = {model.noise: n,
                                            keep_prob: 1.0,
                                            is_training:False})
            
            #Here we will plot all the images of a single batch in montage form
            imgs = [img[:,:,0] for img in gen_img]
            m = montage(imgs)
            gen_img = m
            plt.axis('off')
            plt.imshow(gen_img, cmap='gray')
            plt.draw()
            plt.pause(0.00001)

def main():
    model = GAN() 
    train(model)
    print('Model Initialized...')

main()                       