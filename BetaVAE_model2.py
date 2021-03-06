import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pickle


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n = 60000
beta = 1

tf.reset_default_graph()

batch_size = 100
n_class = 10
X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X_in')
X_out = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X_out')
X_out_flat = tf.reshape(X_out, shape=[-1, 28*28])
y = tf.placeholder(dtype=tf.float32, shape=[None, n_class], name='labels')
keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name='keep_prob')

dec_in_channels = 1
n_latent = 10
n_hidden1 = 100
n_hidden2 = 50
img_dim = 28
flatten = 7*7*16
reshaped_dim = [-1, 7, 7, dec_in_channels]
inputs_decoder = int(49 * dec_in_channels / 2)


def initialize_uninitialized_vars(sess):
    from itertools import compress
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var)) \
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))


def encoder(X_in, dim1_input, dim2_input, keep_prob):
    with tf.variable_scope("encoder", reuse=None):
        X = tf.reshape(X_in, shape=[-1, dim1_input, dim2_input, 1])
        x = tf.layers.conv2d(X, filters=64, kernel_size=4, strides=2, padding='same',
                             activation=tf.nn.leaky_relu, name='conv1')
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=2, padding='same',
                             activation=tf.nn.leaky_relu, name='conv2')
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d(x, filters=64, kernel_size=4, strides=1, padding='same',
                             activation=tf.nn.leaky_relu, name='conv3')
        x = tf.nn.dropout(x, keep_prob)
        x = tf.contrib.layers.flatten(x)
        mn = tf.layers.dense(x, units=n_latent, name='dense_mn')
        log_sd = tf.layers.dense(x, units=n_latent, name='dense_sd')
        epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
        z = mn + tf.multiply(epsilon, tf.exp(log_sd))
        z = tf.identity(z, name='sampled')
        mn = tf.identity(z, name='mn')
        log_sd = tf.identity(z, name='sd')

        return z, mn, log_sd


def decoder(sampled_z, y, keep_prob):
    with tf.variable_scope("decoder", reuse=None):
        new_z = tf.concat((sampled_z, y), axis=1)
        x = tf.layers.dense(new_z, units=inputs_decoder, activation=tf.nn.leaky_relu, name='dense1')
        x = tf.layers.dense(x, units=inputs_decoder * 2 + 1, activation=tf.nn.leaky_relu, name='dense2')
        x = tf.reshape(x, reshaped_dim)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=2, padding='same',
                                       activation=tf.nn.leaky_relu, name='deconv1')
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                       activation=tf.nn.leaky_relu, name='deconv2')
        x = tf.nn.dropout(x, keep_prob)
        x = tf.layers.conv2d_transpose(x, filters=64, kernel_size=4, strides=1, padding='same',
                                       activation=tf.nn.leaky_relu, name='deconv3')

        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, units=28 * 28, activation=tf.nn.sigmoid, name='dense3')
        img = tf.reshape(x, shape=[-1, 28, 28], name="img_rec")

        return img


def classifier(z, n_input, n_hidden1, n_hidden2, n_output, keep_prob):
    with tf.variable_scope("classifier", reuse=False):
        # 1st hidden layer
        w03 = tf.Variable(tf.truncated_normal(shape=(n_input, n_hidden1), mean=0, stddev=0.1), name='w03')
        b03 = tf.Variable(tf.zeros(n_hidden1), name='b03')
        h03 = tf.matmul(z, w03) + b03
        h03 = tf.nn.leaky_relu(h03, name='dense1')
        h03 = tf.nn.dropout(h03, keep_prob)

        w13 = tf.Variable(tf.truncated_normal(shape=(n_hidden1, n_hidden2), mean=0, stddev=0.1), name='w13')
        b13 = tf.Variable(tf.zeros(n_hidden2), name='b13')
        h13 = tf.matmul(h03, w13) + b13
        h13 = tf.nn.leaky_relu(h13, name='dense2')
        h13 = tf.nn.dropout(h13, keep_prob)

        # output layer
        wo3 = tf.Variable(tf.truncated_normal(shape=(n_hidden2, n_output), mean=0, stddev=0.1), name='wo3')
        bo3 = tf.Variable(tf.zeros(10), name='bo3')
        logits = tf.matmul(h13, wo3) + bo3
        logits = tf.identity(logits, name='logits')

    return logits


sampled, mu, log_sigma = encoder(X_in, img_dim, img_dim, keep_prob)
rec_img = decoder(sampled, y, keep_prob)

X_out_reshape = tf.reshape(rec_img, [-1, 28*28])
X_in_reshape = tf.reshape(X_in, [-1, 28*28])
marginal_likelihood = -tf.reduce_sum(X_in_reshape * tf.log(X_out_reshape) + (1 - X_in_reshape) * tf.log(1 - X_out_reshape), 1)
KL_divergence = -0.5 * tf.reduce_sum(-tf.square(mu) - tf.exp(2*log_sigma) + 2*log_sigma + 1, 1)
marginal_likelihood = tf.reduce_mean(marginal_likelihood)
KL_divergence = tf.reduce_mean(KL_divergence)
ELBO = marginal_likelihood + beta*KL_divergence
loss = ELBO
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

saver = tf.train.Saver()

ploss = []
platent_loss = []
paccu = []
prec_loss = []
pclass_loss = []


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(6000):
        batch = mnist.train.next_batch(batch_size=batch_size)

        batch_x = [np.reshape(b, [28, 28]) for b in batch[0]]
        batch_y = batch[1]

        _, tot_loss, loss_likelihood, loss_divergence = \
            sess.run((train_op, loss, marginal_likelihood, KL_divergence),
                     feed_dict={X_in: batch_x, X_out: batch_x, y: batch_y, keep_prob: 0.75})

        ploss.append(tot_loss)
        platent_loss.append(loss_divergence)
        prec_loss.append(loss_likelihood)

        print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f " % (
        epoch, tot_loss, -loss_likelihood, loss_divergence))

    saver.save(sess, 'Bu1/vae1')
    print("Model saved")

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('Bu1/vae1.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('Bu1'))
    vae_graph = tf.get_default_graph()

    new_mu = vae_graph.get_tensor_by_name('encoder/mn:0')
    new_sd = vae_graph.get_tensor_by_name('encoder/sd:0')
    new_mu_sg = tf.stop_gradient(new_mu)
    new_sd_sg = tf.stop_gradient(new_sd)
    class_input = tf.concat((new_mu_sg, new_sd_sg), axis=1)

    logits = classifier(class_input, 2*n_latent, n_hidden1, n_hidden2, 10, keep_prob)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    training_operation = optimizer.minimize(loss_operation)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    initialize_uninitialized_vars(sess)

    for epoch in range(10000):
        batch = mnist.train.next_batch(batch_size=batch_size)
        batch_x = [np.reshape(b, [28, 28]) for b in batch[0]]
        batch_y = batch[1]
        if epoch % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={X_in: batch_x, y: batch_y, keep_prob: 1.0})
            paccu.append(train_accuracy)
            print('step %d, training accuracy %g' % (epoch, train_accuracy))
        _, loss1 = sess.run((training_operation, loss_operation), feed_dict={X_in: batch_x, y: batch_y, keep_prob: 0.75})
        pclass_loss.append(loss1)


with open('resultBu1.pckl','wb') as f:
    pickle.dump([paccu, ploss, platent_loss, prec_loss, pclass_loss], f)