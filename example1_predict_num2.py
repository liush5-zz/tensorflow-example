#!/usr/bin/Python
# -*- coding: utf-8 -*-


"""A simple MNIST classifier: Predict handwriting number ---step 2

This script is based on the Tensoflow MNIST beginners tutorial
See extensive documentation for the tutorial at
https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
"""

#import modules
import sys
import tensorflow as tf
from PIL import Image, ImageFilter
import numpy as np

def predictint(imvalue):
    """
    returns a predicted integer.
    """
    
    # Define the model (same as when creating the model file)
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)
    
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)
       
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')   
    
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    
    x_image = tf.reshape(x, [-1,28,28,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    
    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    saver = tf.train.Saver()
    

    with tf.Session() as sess:

        saver.restore(sess, "tmp_mnist/model.ckpt")
        print ("Model restored.")
       
        prediction=tf.argmax(y_conv,1)
        return prediction.eval(feed_dict={x: imvalue ,keep_prob: 1.0}, session=sess)




def imageprepare(argv):
    """
    This function returns a numpy values.
    """
    im = Image.open(argv).convert('L')
    img = im.resize((28, 28), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    data = img.getdata()

    data = np.matrix(data,dtype="float")
    data = (255.0 - data) / 255.0

    new_data = np.reshape(data, (1, 28 * 28))
    return new_data




def main(argv=None):
    """
    Main function.
    """
    path = "map/testmap5.jpg"
    
    imvalue = imageprepare(path)
    
    imvalue = np.array(imvalue)
    print("imvalue.shape:",imvalue.shape)
    print("----------------------------")
    predint = predictint(imvalue)
    print ("result:",predint[0]) 
    
if __name__ == "__main__":
    main()
