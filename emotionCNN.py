import os
import numpy as np
import tensorflow as tf
from random import seed
from random import shuffle
from sys import stdout
from datetime import datetime
import tensorflow as tf
from tensorflow.python.framework import ops

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
    return np.array([1,0,0,0,0])
  elif emotion == 'SA':
    return np.array([0,1,0,0,0])
  elif emotion == 'AN':
    return np.array([0,0,1,0,0])
  elif emotion == 'DI':
        return np.array([0,0,0,1,0])
  elif emotion == 'FE':
        return np.array([0,0,0,0,1])
  else:
    raise AssertionError("Unable to determine emotion from name \"%s\"." % name)

def log(f, string):
  print string
  f.write('%s\n' % string)


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

def train_net(training_set, testing_set, validation_set, fold_index, logFile):
  ops.reset_default_graph()
  tf.set_random_seed(9002)
  sess = tf.Session()
  with open(logFile, 'a') as f:
    log(f, 'Validation Labels')
    log(f, np.array_str(validation_set[1]))
    log(f, 'Testing labels')
    log(f, np.array_str(testing_set[1]))
    log(f, 'Training Size')
    log(f, len(training_set[0]))
    #input nodes
    x = tf.placeholder(tf.float32, shape=[None, 65536]) #256*256
    number_of_labels = 5
    #output nodes
    y_ = tf.placeholder(tf.float32, shape=[None, number_of_labels])

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
    good_validation_accuracy_threshold = 0.85
    good_validation_count_threshold = 5
    for k in range(50000):
      #training is done in batches of 5 images at a time
      i =  (k * 5) % (len(training_set[0]) - 5)
      #After exhausting all training images shuffle and do more batching
      if (i == 0):
        shuffled = [[training_set[0][l], training_set[1][l]] for l in range(len(training_set[0]))]
        shuffle(shuffled)
        training = [[row[0] for row in shuffled], [row[1] for row in shuffled]]
      if k%250 == 0:
        #evaluate accuracy against training set
        train_accuracy = 0
        for j in range(len(training_set[0])):
          train_accuracy += accuracy.eval(session=sess, feed_dict={x: training_set[0][j:j+1], y_: training_set[1][j:j+1], keep_prob: 1.0})
        log(f, "%d, %d, training accuracy, %g"%(fold_index, k, train_accuracy / len(training_set[0])))
        # #prediction of ith image
        # log(f, '\t%s' % np.array_str(y_conv.eval(session=sess, feed_dict={x: training_set[0][i:i+1], keep_prob: 1.0})))
        # #actual label
        # log(f, '\t%s' % np.array_str(labels[i]))
      if k%500 == 0:
        #evaluate accuracy against validation set
        validation_accuracy = 0
        for j in range(len(validation_set[0])):
          validation_accuracy += accuracy.eval(session=sess, feed_dict={x: validation_set[0][j:j+1], y_: validation_set[1][j:j+1], keep_prob: 1.0})
        validation_accuracy = validation_accuracy / len(validation_set[0])
        log(f, "%d, %d, validation accuracy, %g"%(fold_index, k, validation_accuracy))
        if validation_accuracy >= good_validation_accuracy_threshold:
          repeated_good_validation_accuracy += 1
        else:
          repeated_good_validation_accuracy = 0;
      # validation set performed good enough enough times so stop training early
      if (repeated_good_validation_accuracy >= good_validation_count_threshold):
        break
      #Training step of 5 images at a time with dropout probability of 0.5
      train_step.run(session=sess, feed_dict={x: training_set[0][i:i+5], y_: training_set[1][i:i+5], keep_prob: 1.0})

    test_accuracy = 0
    for i in range(len(testing_set[0])):
      log(f, np.array_str(y_conv.eval(session=sess, feed_dict={x: testing_set[0][i:i+1], keep_prob: 1.0})))
      log(f, np.array_str(testing_set[1][i:i+1]))
      test_accuracy += accuracy.eval(session=sess, feed_dict={x: testing_set[0][i:i+1], y_: testing_set[1][i:i+1], keep_prob: 1.0})

    log(f, "%d, test accuracy, %g"%(fold_index, test_accuracy/len(testing[0])))


filenames = []
labels = []
images = []
sess = tf.Session()
for root, dirs, files in os.walk('jaffetest', topdown=False):
  shuffle(files)
  for name in files:
    if name[3:5] != 'HA' and name[3:5] != 'SA' and name[3:5] != 'AN' and name[3:5] != 'FE' and name[3:5] != 'DI':
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

print images.shape[0]
logFile = "logs/%s.log" % datetime.strftime(datetime.now(), '%Y%m%d-%H-%M-%S')

training = [images[40:], labels[40:]]
testing = [images[:20], labels[:20]]
validation = [images[20:40], labels[20:40]]
train_net(training, testing, validation, 1, logFile)

training = [np.concatenate((images[:20],images[60:]), axis=0), np.concatenate((labels[:20],labels[60:]), axis=0)]
testing = [images[20:40], labels[20:40]]
validation = [images[40:60], labels[40:60]]
train_net(training, testing, validation, 2, logFile)

training = [np.concatenate((images[:40],images[80:]), axis=0), np.concatenate((labels[:40],labels[80:]), axis=0)]
testing = [images[40:60], labels[40:60]]
validation = [images[60:80], labels[60:80]]
train_net(training, testing, validation, 3, logFile)

training = [np.concatenate((images[:60],images[100:]), axis=0), np.concatenate((labels[:60],labels[100:]), axis=0)]
testing = [images[60:80], labels[60:80]]
validation = [images[80:100], labels[80:100]]
train_net(training, testing, validation, 0, logFile)
