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
	else:
		raise AssertionError('Unable to determine emotion from name \"%s\".' % name)

import tensorflow as tf
sess = tf.Session()
sess = tf.Session()
filenames = []
labels = []
images = []
for root, dirs, files in os.walk('jaffetest', topdown=False):
	shuffle(files)
	for name in files:
		if os.path.splitext(os.path.join(root, name))[1].lower() == ".png":
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
sess.run(tf.initialize_all_variables())
y = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
for i in range(images.shape[0] - 10):
 	sess.run(train_step, feed_dict={x: images[i:i+1], y_: labels[i:i+1]})
	if i%5 == 0:
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		train_accuracy = accuracy.eval(session=sess, feed_dict={x:images[:i], y_: labels[:i]})
		print("step %d, training accuracy %g"%(i, train_accuracy))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(session=sess, feed_dict={x: images[-10:], y_: labels[-10:]}))


