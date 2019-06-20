#########
# movielens dataset for testing. using multiple gpus.
# user, movie, rating, tags
#########
import tensorflow as tf
import pandas as pd
from tensorflow.python.client import device_lib
import numpy as np

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


print(get_available_gpus())
print(tf.__version__),

n = 100
p = 100
k = 10

x = tf.placeholder('float', shape=[None, p])
y = tf.placeholder('float', shape=[None, 1])
w0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.zeros([p]))

v = tf.Variable(tf.random_normal([k, p], stddev=0.01))

y_hat = tf.Variable(tf.zeros([n, 1]))

linear_term = tf.add(w0, tf.reduce_sum(tf.multiply(w, x), 1, keepdims=True))

pair_interactions = tf.multiply(0.5, tf.reduce_sum(
    tf.subtract(tf.pow(tf.matmul(x, tf.transpose(v)), 2), tf.matmul(tf.pow(x, 2), tf.transpose(tf.pow(v, 2)))
                ), 1, keepdims=True))
y_hat=tf.add(linear_term,pair_interactions)

lambda_w=tf.constant(0.001,name='lambda_w')
lambda_v=tf.constant(0.001,name='lambda_v')
l2_norm=tf.reduce_sum(tf.add(tf.multiply(lambda_w,tf.pow(w,2)),tf.multiply(lambda_v,tf.pow(v,2))))
error=tf.reduce_mean(tf.square(tf.subtract(y,y_hat)))
loss=tf.add(error,l2_norm)

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)






