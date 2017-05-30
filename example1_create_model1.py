#!/usr/bin/Python
# -*- coding: utf-8 -*-


"""A simple MNIST classifier: Predict handwriting number ---step 1

This script is based on the Tensoflow MNIST beginners tutorial
See extensive documentation for the tutorial at
https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
"""

#import modules
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#import data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]), name="w")
b = tf.Variable(tf.zeros([10]), name="b")
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init_op = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())


# Train and save the model: model.ckpt file
with tf.Session() as sess:
    sess.run(init_op)
    
    for i in range(10000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
                
    save_path = saver.save(sess, "tmp_mnist/model.ckpt")
    print ("Model saved in file: ", save_path)

