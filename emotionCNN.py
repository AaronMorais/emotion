import os
from PIL import Image
import numpy as np
import tensorflow as tf
from random import shuffle

def label_from_name(name):
	emotion = name[3:5]
	if emotion == 'AN':
		return np.array([1,0,0,0,0,0,0])
	elif emotion == 'DI':
		return np.array([0,1,0,0,0,0,0])
	elif emotion == 'FE':
		return np.array([0,0,1,0,0,0,0])
	elif emotion == 'HA':
		return np.array([0,0,0,1,0,0,0])
	elif emotion == 'NE':
		return np.array([0,0,0,0,1,0,0])
	elif emotion == 'SA':
		return np.array([0,0,0,0,0,1,0])
	elif emotion == 'SU':
		return np.array([0,0,0,0,0,0,1])

import tensorflow as tf
sess = tf.Session()
sess = tf.Session()
filenames = []
labels = []
images = []
for root, dirs, files in os.walk('jaffetest', topdown=False):
	shuffle(files)
	for name in files:
		labels += [label_from_name(name)]
		images += [tf.image.decode_png(tf.read_file(os.path.join(root, name)), channels=1).eval(session=sess)]

images = np.array(images)
# Convert from [0, 255] -> [0.0, 1.0].
images = images.astype(np.float32)
images = np.multiply(images, 1.0 / 255.0)
labels = np.array(labels, dtype='float32')
assert images.shape[3] == 1
images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])
x = tf.placeholder(tf.float32, shape=[None, 65536]) #256*256
number_of_labels = 7
y_ = tf.placeholder(tf.float32, shape=[None, number_of_labels])
W = tf.Variable(tf.truncated_normal([65536,number_of_labels], stddev=0.001))
b = tf.Variable(tf.truncated_normal([number_of_labels], stddev=0.001))
# sess.run(tf.initialize_all_variables())
# y = tf.nn.softmax(tf.matmul(x,W) + b)
# cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# for i in range(images.shape[0] - 10):
#  	sess.run(train_step, feed_dict={x: images[i:i+1], y_: labels[i:i+1]})
# 	if i%10 == 0:
# 		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 		train_accuracy = accuracy.eval(session=sess, feed_dict={x:images[:i], y_: labels[:i]})
# 		print("step %d, training accuracy %g"%(i, train_accuracy))

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(session=sess, feed_dict={x: images[-10:], y_: labels[-10:]}))

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

W_conv1 = weight_variable([10, 10, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,256,256,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([10, 10, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# W_conv3 = weight_variable([10, 10, 64, 128])
# b_conv3 = bias_variable([128])

# h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool3 = max_pool_2x2(h_conv2)

# W_conv4 = weight_variable([10, 10, 128, 256])
# b_conv4 = bias_variable([256])

# h_conv4 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
# h_pool4 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([64 * 64 * 64, 256])
b_fc1 = bias_variable([256])

h_pool2_flat = tf.reshape(h_pool2, [-1, 64*64*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([256, number_of_labels])
b_fc2 = bias_variable([number_of_labels])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv+ 1e-9))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
# for i in range(images.shape[0] - 10):
#  	sess.run(train_step, feed_dict={x: images[i:i+1], y_: labels[i:i+1]})
# 	if i%10 == 0:
# 		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 		train_accuracy = accuracy.eval(session=sess, feed_dict={x:images[:i], y_: labels[:i]})
# 		print("step %d, training accuracy %g"%(i, train_accuracy))

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(accuracy.eval(session=sess, feed_dict={x: images[-10:], y_: labels[-10:]}))
for i in range(images.shape[0] - 10):
	if i%20 == 0:
		train_accuracy = 0
		for j in range(i):
			train_accuracy += accuracy.eval(session=sess, feed_dict={x: images[j:j+1], y_: labels[j:j+1], keep_prob: 1.0})
		print("step %d, training accuracy %g"%(i, train_accuracy / float(i + 1)))
	train_step.run(session=sess, feed_dict={x: images[i:i+1], y_: labels[i:i+1], keep_prob: 0.5})

test_accuracy = 0
for i in range(10):
	test_accuracy += accuracy.eval(session=sess, feed_dict={x: images[-i:], y_: labels[-i:], keep_prob: 1.0})
print("test accuracy %g"%(test_accuracy / 10.0))

