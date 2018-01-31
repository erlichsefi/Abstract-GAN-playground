"""
This is a straightforward Python implementation of a generative adversarial network.
The code is drawn directly from the O'Reilly interactive tutorial on GANs
(https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners).

This is an genral implemention of the GANs
"""

import tensorflow as tf
import numpy as np
import datetime
import cv2,os
from tensorflow.examples.tutorials.mnist import input_data

# consts
inputfolder="photo_database_working_folder/"
image_width=10
image_higth=10
image_channel=3
genrtor_input_size=int(image_width*image_higth*image_channel*1.13)
batch_size = 10

# read the images
onlyfiles = [os.path.join(inputfolder, f) for f in os.listdir(inputfolder) if os.path.isfile(os.path.join(inputfolder, f))]
input_images =[cv2.imread(m).reshape(image_width*image_higth*image_channel) for m in onlyfiles] # will be m X image_width*image_higth*image_channel

'''
Return a total of `num` random samples and labels.
'''
def next_batch(num, data):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    return np.asarray(data_shuffle)


# Define the discriminator network
# images size - [None,image_width,image_higth,image_channel]
def discriminator(images, reuse_variables=None):

    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables) as scope:

	# input - [b_m,image_width,image_higth,image_channel]
	# First convolutional and pool layers
        d_w1 = tf.get_variable('d_w1', [5, 5, image_channel, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
        d1 = tf.nn.conv2d(input=images, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')

	#  [b_m,image_width,image_higth,32]
        d1 = d1 + d_b1
        d1 = tf.nn.relu(d1)
        d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#  [b_m,image_width/2,image_higth/2,32]

        # Second convolutional and pool layers
        # This finds 64 different 5 x 5 pixel features
        d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
        d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
	#  [b_m,image_width/2,image_higth/2,64]
        d2 = d2 + d_b2
        d2 = tf.nn.relu(d2)
        d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	#  output  [b_m,image_width/4,image_higth/4,64]

        # First fully connected layer
        d=int((image_width/4)*(image_higth/4)*64)
        full_connected_layer_size=int(d/3)

        d_w3 = tf.get_variable('d_w3', [d, full_connected_layer_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b3 = tf.get_variable('d_b3', [full_connected_layer_size], initializer=tf.constant_initializer(0))
        d3 = tf.reshape(d2, [-1, d])
        d3 = tf.matmul(d3, d_w3)
        d3 = d3 + d_b3
        d3 = tf.nn.relu(d3)

        # Second fully connected layer
        d_w4 = tf.get_variable('d_w4', [full_connected_layer_size, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
        d_b4 = tf.get_variable('d_b4', [1], initializer=tf.constant_initializer(0))
        d4 = tf.matmul(d3, d_w4) + d_b4

        # d4 contains unscaled values
        return d4

# Define the generator network
# z size = [None, random_size] (the random vectors)
# batch_size - number of images to genrate
# random_dim - the dim of the random vector
def generator(z, batch_size, random_dim):

    #
    first_layer_size=2*image_width*image_higth*image_channel
    g_w1 = tf.get_variable('g_w1', [random_dim, first_layer_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [first_layer_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.matmul(z, g_w1) + g_b1
    # from a random vector genrate
    g1 = tf.reshape(g1, [-1, image_width, image_higth, image_channel])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='bn1')
    g1 = tf.nn.relu(g1)
    #  [-1,image_width,image_higth,image_channel]

    # Generate 50 features
    g_w2 = tf.get_variable('g_w2', [3, 3, image_channel, int(random_dim/2)], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b2 = tf.get_variable('g_b2', [int(random_dim/2)], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g2 = tf.nn.conv2d(g1, g_w2, strides=[1, 2, 2, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=1e-5, scope='bn2')
    g2 = tf.nn.relu(g2)
    g2 = tf.image.resize_images(g2, [image_width*2, image_higth*2])

    # Generate 25 features
    g_w3 = tf.get_variable('g_w3', [3, 3, int(random_dim/2), int(random_dim/4)], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b3 = tf.get_variable('g_b3', [int(random_dim/4)], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g3 = tf.nn.conv2d(g2, g_w3, strides=[1, 2, 2, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=1e-5, scope='bn3')
    g3 = tf.nn.relu(g3)
    g3 = tf.image.resize_images(g3, [image_width, image_higth*image_channel])

    # Final convolution with one output channel
    g_w4 = tf.get_variable('g_w4', [1, 1, int(random_dim/4), 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b4 = tf.get_variable('g_b4', [1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g4 = tf.nn.conv2d(g3, g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4
    g4 = tf.sigmoid(g4)

    g4 = tf.reshape(g4, [-1, image_width, image_higth, image_channel])
    # Dimensions of g4: batch_size x image_width x image_higth x image_channel
    return g4



z_placeholder = tf.placeholder(tf.float32, [None, genrtor_input_size], name='z_placeholder')
# z_placeholder is for feeding input noise to the generator

x_placeholder = tf.placeholder(tf.float32, shape = [None,image_width,image_higth,image_channel], name='x_placeholder')
# x_placeholder is for feeding input images to the discriminator

Gz = generator(z_placeholder, batch_size, genrtor_input_size)
# Gz holds the generated images

Dx = discriminator(x_placeholder)
# Dx will hold discriminator prediction probabilities
# for the real MNIST images

Dg = discriminator(Gz, reuse_variables=True)
# Dg will hold discriminator prediction probabilities for generated images

# Define losses
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

# Define variable lists
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]
g_vars = [var for var in tvars if 'g_' in var.name]

# Define the optimizers
# Train the discriminator
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

# Train the generator
g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

# From this point forward, reuse variables
tf.get_variable_scope().reuse_variables()

sess = tf.Session()

# Send summary statistics to TensorBoard
tf.summary.scalar('Generator_loss', g_loss)
tf.summary.scalar('Discriminator_loss_real', d_loss_real)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

images_for_tensorboard = generator(z_placeholder, batch_size, genrtor_input_size)
tf.summary.image('Generated_images', images_for_tensorboard, 5)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

sess.run(tf.global_variables_initializer())




# Pre-train discriminator
for i in range(300):
    z_batch = np.random.normal(0, 1, size=[batch_size, genrtor_input_size])
    real_image_batch = next_batch(batch_size,mnist).reshape([batch_size, image_width, image_higth, image_channel])
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

# Train generator and discriminator together
for i in range(100000):
    real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, image_width, image_higth, image_channel])
    z_batch = np.random.normal(0, 1, size=[batch_size, genrtor_input_size])

    # Train discriminator on both real and fake images
    _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                           {x_placeholder: real_image_batch, z_placeholder: z_batch})

    # Train generator
    z_batch = np.random.normal(0, 1, size=[batch_size, genrtor_input_size])
    _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

    if i % 10 == 0:
        # Update TensorBoard with summary statistics
        z_batch = np.random.normal(0, 1, size=[batch_size, genrtor_input_size])
        summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
        writer.add_summary(summary, i)
