# -*- coding: utf-8 -*-

# Sample code to use string producer.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h
    """
    o_h = np.zeros(n)
    o_h[x] = 1.
    return o_h

num_classes = 3
batch_size = 4

# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------
def dataSource(paths, batch_size):

    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []
    labels = []
    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), [i] if num_classes <= 2 else one_hot(i, num_classes)
        image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                          min_after_dequeue=min_after_dequeue)
        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)

    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch

# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------
def myModel(X, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch_size * 3, 18 * 33 * 64]), units=30, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)

    return y

example_batch_train, label_batch_train = dataSource(["data_practica3/train1/*.jpg", "data_practica3/train2/*.jpg", "data_practica3/train3/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["data_practica3/valid1/*.jpg", "data_practica3/valid2/*.jpg", "data_practica3/valid3/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["data_practica3/test1/*.jpg", "data_practica3/test2/*.jpg", "data_practica3/test3/*.jpg"], batch_size=batch_size)

example_batch_train_predicted = myModel(example_batch_train, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, reuse=True)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - tf.cast(label_batch_train, dtype=tf.float32)))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - tf.cast(label_batch_valid, dtype=tf.float32)))
cost_test = tf.reduce_sum(tf.square(example_batch_test_predicted - tf.cast(label_batch_test, dtype=tf.float32)))

y = tf.placeholder(tf.float32, [None, 3])
y_ = tf.placeholder(tf.float32, [None, 3])
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.
errorTrain = []
errorValid = []
epoch = 0

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    accuracyBefore = 0
    accuracyAfter = 0
    mejora = 1

    while (mejora > 0.05 or accuracyAfter < 0.90) and epoch < 100:
        epoch += 1
        sess.run(optimizer)

        accuracyAfter = sess.run(accuracy, feed_dict={y:example_batch_train_predicted.eval(), y_:label_batch_train.eval()})
        mejora = np.absolute(accuracyAfter - accuracyBefore)
        accuracyBefore = accuracyAfter

        print("Iter:", epoch, "---------------------------------------------")
        #print(sess.run(label_batch_valid))
        #print(sess.run(example_batch_valid_predicted))

        aux_train = sess.run(cost)
        aux_valid = sess.run(cost_valid)

        print("Error de entrenamiento:", aux_train)
        print("Error de validación:", aux_valid)
        print("Precisión de entrenamiento: ", accuracyAfter)
        print("Precisión de validación: ", sess.run(accuracy, feed_dict={y:example_batch_valid_predicted.eval(), y_: label_batch_valid.eval()}))

        errorTrain.append(aux_train)
        errorValid.append(aux_valid)

    save_path = saver.save(sess, "./tmp/model.ckpt")
    print("Model saved in file: %s" % save_path)
    print("Error de test: ", sess.run(cost_test))
    print("Precisión de test: ",
          sess.run(accuracy, feed_dict={y: example_batch_test_predicted.eval(), y_: label_batch_test.eval()}))

    coord.request_stop()
    coord.join(threads)

plt.figure()
plt.plot(errorTrain, 'b', linewidth = 2, label = 'Train')
plt.plot(errorValid, 'r', linewidth = 2, label = 'Valid')
plt.xlabel("Iteraciones", fontsize=20)
plt.ylabel("Error", fontsize=20)
plt.legend()
plt.show()

"""
76 iteraciones, 5 neuronas: error train/valid/test: 3.404906/5.831165/6.7976823; precisión train/valid/test: 1.0/0.5/0.83
34 iteraciones, 10 neuronas: error train/valid/test: 0.5181379/4.3331785/4.2009907; precisión train/valid/test: 1.0/0.83/0.92
28 iteraciones, 15 neuronas: error train/valid/test: 2.8894546/3.2216973/3.5537062; precisión train/valid/test: 1.0/1.0/0.83
30 iteraciones, 20 neuronas: error train/valid/test: 2.0864758/3.4267962/3.77769; precisión train/valid/test: 1.0/0.83/0.75
30 iteraciones, 30 neuronas: error train/valid/test: 0.6253191/2.297392/3.0339146; precisión train/valid/test: 1.0/1.0/0.92
11 iteraciones, 50 neuronas: error train/valid/test: 1.4662057/3.3126972/3.684247; precisión train/valid/test: 1.0/0.83/0.92
12 iteraciones, 80 neuronas: error train/valid/test: 1.6972865/5.0280867/2.1076648; precisión train/valid/test: 1.0/0.83/0.92
"""