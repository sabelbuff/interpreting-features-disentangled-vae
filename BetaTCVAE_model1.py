import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import pickle


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
n = 60000
beta = 1
alpha = 1

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


def TCpenalty1(mu, log_sigma):
    with tf.variable_scope("TCpenalty1", reuse=False):
        B = tf.constant(batch_size, dtype=tf.float32)
        N = tf.constant(n, dtype=tf.float32)
        sigma = tf.exp(log_sigma)
        mu1 = mu[0]
        sigma1 = sigma[0]
        mu_rest1 = mu[1:tf.shape(mu)[0]-1]
        sigma_rest1 = sigma[1:tf.shape(sigma)[0]-1]
        mu_last = mu[tf.shape(mu)[0]-1]
        sigma_last = sigma[tf.shape(sigma)[0]-1]
        dist1 = tf.distributions.Normal(loc=mu1, scale=sigma1)
        z1 = dist1.sample([1])
        prob_z1 = dist1.prob(z1)
        temp1 = tf.divide(tf.reduce_prod(prob_z1), N)
        dist2 = tf.distributions.Normal(loc=mu_rest1, scale=sigma_rest1)
        z_rest1 = dist2.sample([1])
        z_rest1 = tf.reshape(z_rest1, [tf.shape(mu_rest1)[0], n_latent])
        prob_z_rest1 = dist2.prob(z_rest1)
        temp2 = tf.divide(tf.reduce_sum(tf.reduce_prod(prob_z_rest1, axis=1)), B)
        dist3 = tf.distributions.Normal(loc=mu_last, scale=sigma_last)
        z_last = dist3.sample([1])
        prob_z_last = dist3.prob(z_last)
        temp3 = tf.multiply(tf.divide(tf.subtract(N, B), tf.multiply(N, B)), tf.reduce_prod(prob_z_last))

        prob_z = tf.add(tf.add(temp1, temp2), temp3)

    return prob_z


def TCpenalty2(mu, log_sigma):
    with tf.variable_scope("TCpenalty2", reuse=False):
        B = tf.constant(batch_size, dtype=tf.float32)
        N = tf.constant(n, dtype=tf.float32)
        sigma = tf.exp(log_sigma)
        mu1 = mu[0]
        sigma1 = sigma[0]
        mu_rest1 = mu[1:tf.shape(mu)[0] - 1]
        sigma_rest1 = sigma[1:tf.shape(sigma)[0] - 1]
        mu_last = mu[tf.shape(mu)[0] - 1]
        sigma_last = sigma[tf.shape(sigma)[0] - 1]
        dist1 = tf.distributions.Normal(loc=mu1, scale=sigma1)
        z1 = dist1.sample([1])
        prob_z1 = dist1.prob(z1)
        temp1 = tf.divide(prob_z1, N)
        dist2 = tf.distributions.Normal(loc=mu_rest1, scale=sigma_rest1)
        z_rest1 = dist2.sample([1])
        z_rest1 = tf.reshape(z_rest1, [tf.shape(mu_rest1)[0], n_latent])
        prob_z_rest1 = dist2.prob(z_rest1)
        temp2 = tf.divide(tf.reduce_sum(prob_z_rest1, axis=0), B)
        dist3 = tf.distributions.Normal(loc=mu_last, scale=sigma_last)
        z_last = dist3.sample([1])
        prob_z_last = dist3.prob(z_last)
        temp3 = tf.multiply(tf.divide(tf.subtract(N, B), tf.multiply(N, B)), prob_z_last)

        prob_z = tf.add(tf.add(temp1, temp2), temp3)
        prob_z_all = tf.reduce_prod(prob_z)

    return prob_z_all


sampled, mu, log_sigma = encoder(X_in, img_dim, img_dim, keep_prob)
rec_img = decoder(sampled, y, keep_prob)
class_input = tf.concat((mu, log_sigma), axis=1)
logits = classifier(class_input, n_latent*2, n_hidden1, n_hidden2, n_class, keep_prob)


X_out_reshape = tf.reshape(rec_img, [-1, 28*28])
X_in_reshape = tf.reshape(X_in, [-1, 28*28])
marginal_likelihood = -tf.reduce_sum(X_in_reshape * tf.log(X_out_reshape) + (1 - X_in_reshape) * tf.log(1 - X_out_reshape), 1)
KL_divergence = -0.5 * tf.reduce_sum(-tf.square(mu) - tf.exp(2*log_sigma) + 2*log_sigma + 1, 1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
classifier_loss = tf.reduce_mean(cross_entropy)
marginal_likelihood = tf.reduce_mean(marginal_likelihood)
KL_divergence = tf.reduce_mean(KL_divergence)
num = TCpenalty1(mu, log_sigma)
dem = TCpenalty2(mu, log_sigma)
Total_correlation = tf.divide(num, dem)
ELBO = marginal_likelihood + KL_divergence
loss = ELBO + beta*Total_correlation + alpha*classifier_loss
train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

ploss = []
platent_loss = []
paccu = []
prec_loss = []
pclass_loss = []
ptc_loss = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(6000):
        batch = mnist.train.next_batch(batch_size=batch_size)

        batch_x = [np.reshape(b, [28, 28]) for b in batch[0]]
        batch_y = batch[1]
        if epoch % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={X_in: batch_x, y: batch_y, keep_prob: 1.0})
            paccu.append(train_accuracy)
            print('step %d, training accuracy %g' % (epoch, train_accuracy))
        # if not i % 200:
        _, tot_loss, loss_likelihood, loss_divergence, toc, closs = \
            sess.run((train_op, loss, marginal_likelihood, KL_divergence, Total_correlation, classifier_loss),
                     feed_dict={X_in: batch_x, X_out: batch_x, y: batch_y, keep_prob: 0.75})

        ploss.append(tot_loss)
        platent_loss.append(loss_divergence)
        prec_loss.append(loss_likelihood)
        pclass_loss.append(closs)
        ptc_loss.append(toc)

        print("epoch %d: L_tot %03.2f L_likelihood %03.2f L_divergence %03.2f cor %03.2f" % (
        epoch, tot_loss, -loss_likelihood, loss_divergence, toc))

    saver.save(sess, 'TC11/vae11')
    print("Model saved")

with open('resultTC11.pckl','wb') as f:
    pickle.dump([paccu, ploss, platent_loss, prec_loss, pclass_loss, ptc_loss], f)
