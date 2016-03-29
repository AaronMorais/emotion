import os
import numpy as np
import tensorflow as tf
from random import seed
from random import shuffle

def label_from_name(name):
  emotion = name[3:5]
  # if emotion == 'AN':
  #       return np.array([1,0,0,0,0,0,0])
  # elif emotion == 'DI':
  #       return np.array([0,1,0,0,0,0,0])
  # elif emotion == 'FE':
  #       return np.array([0,0,1,0,0,0,0])
  # elif emotion == 'HA':
  #       return np.array([0,0,0,1,0,0,0])
  # elif emotion == 'NE':
  #       return np.array([0,0,0,0,1,0,0])
  # elif emotion == 'SA':
  #       return np.array([0,0,0,0,0,1,0])
  # elif emotion == 'SU':
  #       return np.array([0,0,0,0,0,0,1])
  if emotion == 'HA':
    return np.array([0,1])
  elif emotion == 'SA':
    return np.array([1,0])
  else:
    raise AssertionError("Unable to determine emotion from name \"%s\"." % name)

import tensorflow as tf
sess = tf.Session()
sess = tf.Session()
filenames = []
labels = []
images = []
seed(9002)
tf.set_random_seed(9002)
for root, dirs, files in os.walk('jaffetest', topdown=False):
	shuffle(files)
	for name in files:
		if name[3:5] != 'HA' and name[3:5] != 'SA':
			continue
		if os.path.splitext(os.path.join(root, name))[1].lower() == '.png':
			labels += [label_from_name(name)]
			images += [tf.image.decode_png(tf.read_file(os.path.join(root, name)), channels=1).eval(session=sess)]

images = np.array(images)
# Convert from [0, 255] -> [0.0, 1.0].
images = images.astype(np.float32)
images = np.multiply(images, 1.0 / 255.0)
labels = np.array(labels, dtype='float32')
# make sure there is only one channel.  i.e. Greyscale not RGB
assert images.shape[3] == 1
images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

training = [images[:-10], labels[:-10]]
testing = [images[-10:], labels[-10:]]
#input nodes
x = tf.placeholder(tf.float32, shape=[None, 65536]) #256*256
number_of_labels = 2
#output nodes
y_ = tf.placeholder(tf.float32, shape=[None, number_of_labels])

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#5x5 convolution window, produces 32 features
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,256,256,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5, 5, 64, 128])
b_conv3 = bias_variable([128])

h_conv3 = tf.sigmoid(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([5, 5, 128, 256])
b_conv4 = bias_variable([256])

h_conv4 = tf.sigmoid(conv2d(h_pool3, W_conv4) + b_conv4)
h_pool4 = max_pool_2x2(h_conv4)

#Original image was 256 * 256, after 4 layers of pooling we have size 16*16
# 2 Fully connected layers of 1024 nodes
W_fc1 = weight_variable([16 * 16 * 256, 1024])
b_fc1 = bias_variable([1024])

h_pool4_flat = tf.reshape(h_pool4, [-1, 16 * 16 * 256])
h_fc1 = tf.sigmoid(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

# Random dropouts to minimize overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, number_of_labels])
b_fc2 = bias_variable([number_of_labels])

#output from graph
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

#Loss equation
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv+ 1e-9))

#parameter relates to the training step size
#smaller values change the net slower
#larger values change the net more drasitically
train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

repeated_good_validation_accuracy = 0
good_validation_accuracy_threshold = 0.99
good_validation_count_threshold = 5
for k in range(0, 10000, 1):
	#training is done in batches of 5 images at a time
	i =  (k * 5) % (len(training[0]) - 5)
	#After exhausting all training images shuffle and do more batching
	if (i == 0):
		shuffled = [[training[0][l], training[1][l]] for l in range(len(training[0]))]
		shuffle(shuffled)
		training = [[row[0] for row in shuffled], [row[1] for row in shuffled]]
	if k%251 == 0:
		#evaluate accuracy against training set
		train_accuracy = 0
		for j in range(len(training[0])):
			train_accuracy += accuracy.eval(session=sess, feed_dict={x: training[0][j:j+1], y_: training[1][j:j+1], keep_prob: 1.0})
		print("step %d %d, training accuracy %g"%(k, i, train_accuracy / float(images.shape[0] - 10)))
		#prediction of ith image
		# print y_conv.eval(session=sess, feed_dict={x: training[0][i:i+1], keep_prob: 1.0})
		#actual label
		# print labels[i]
	if k%502 == 0:
		#evaluate accuracy against validation set
		test_accuracy = 0
		for j in range(len(testing[0])):
			test_accuracy += accuracy.eval(session=sess, feed_dict={x: testing[0][j:j+1], y_: testing[1][j:j+1], keep_prob: 1.0})
		test_accuracy = test_accuracy / len(testing[0])
		print("step %d: test accuracy %g"%(k, test_accuracy))
		if test_accuracy >= good_validation_accuracy_threshold:
			repeated_good_validation_accuracy += 1
		else:
			repeated_good_validation_accuracy = 0;
	# validation set performed good enough enough times so stop training early
	#TODO use an actual validation for early end, not the test set
	if (repeated_good_validation_accuracy >= good_validation_count_threshold):
		break
	#Training step of 5 images at a time with dropout probability of 0.5
	train_step.run(session=sess, feed_dict={x: training[0][i:i+5], y_: training[1][i:i+5], keep_prob: 0.50})

test_accuracy = 0
for i in range(len(testing[0])):
	print y_conv.eval(session=sess, feed_dict={x: testing[0][i:i+1], keep_prob: 1.0})
	print testing[1][i:i+1]
	test_accuracy += accuracy.eval(session=sess, feed_dict={x: testing[0][i:i+1], y_: testing[1][i:i+1], keep_prob: 1.0})

print("test accuracy %g"%(test_accuracy/len(testing[0])))
