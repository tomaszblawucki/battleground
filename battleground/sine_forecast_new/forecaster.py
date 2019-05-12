#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


tf.reset_default_graph()
tf.set_random_seed(42)


t_min, t_max = 0, 30
resolution = 0.1

def time_series(t):
#    return t
    return t * np.sin(t) / 3 + 2 * np.sin(t*5)

def next_batch(batch_size, n_steps):
    t0 = np.arange(0., 1., 1/batch_size)*(t_max - t_min - n_steps * resolution)
    t0 = t0.reshape(-1, 1)
    Ts = t0 + np.arange(0., n_steps + 1) * resolution
    ys = time_series(Ts)
    return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)

t = np.linspace(t_min, t_max, int((t_max - t_min) / resolution))

n_steps = 20
t_instance = np.linspace(12.2, 12.2 + resolution * (n_steps + 1), n_steps + 1)

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.title("Szereg czasowy (wygenerowany)", fontsize=14)
plt.plot(t, time_series(t), label=r"$t . \sin(t) / 3 + 2 . \sin(5t)$")
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "y-", linewidth=3, label="próbka ucząca")
plt.legend(loc="lower left", fontsize=14)
#plt.axis([0, 30, -17, 13])
plt.xlabel("Czas")
plt.ylabel("Wartość")

plt.subplot(122)
plt.title("Próbka ucząca", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="próbka")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "y*", markersize=10, label="w. docelowa")
plt.legend(loc="upper left")
plt.xlabel("Czas")

plt.show()

# =============================================================================
# RNN BUILD
# =============================================================================

n_steps = 20
n_inputs = 1
n_neurons = 512
n_outputs = 1
n_layers = 1

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
cell = tf.nn.rnn_cell.MultiRNNCell(
            cells=[tf.nn.rnn_cell.LSTMCell(num_units=n_neurons, activation=tf.nn.tanh) for _ in range(n_layers)]
)
rnn_outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

n_outputs = 1
learning_rate = 0.001

stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
outputs = tf.reshape(stacked_outputs, [-1, n_steps, n_outputs])

loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_iterations = 1000
batch_size = 50


with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        X_batch, y_batch = next_batch(batch_size, n_steps)
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tBłąd MSE:", mse)
    
    X_new = time_series(np.array(t_instance[:-1].reshape(-1, n_steps, n_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})
    
    saver.save(sess, "./moj_model_szeregow_czasowych")
    
    
plt.title("Testowanie modelu", fontsize=14)
plt.plot(t_instance[:-1], time_series(t_instance[:-1]), "bo", markersize=10, label="próbka")
plt.plot(t_instance[1:], time_series(t_instance[1:]), "y*", markersize=10, label="w. docelowa")
plt.plot(t_instance[1:], y_pred[0,:,0], "r.", markersize=10, label="prognoza")
plt.legend(loc="upper left")
plt.xlabel("Czas")

plt.show()

# =============================================================================
# GENERATOR
# =============================================================================

with tf.Session() as sess:
    saver.restore(sess, "./moj_model_szeregow_czasowych")

    sequence1 = [0. for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence1[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence1.append(y_pred[0, -1, 0])

    sequence2 = [time_series(i * resolution + t_min + (t_max-t_min/3)) for i in range(n_steps)]
    for iteration in range(len(t) - n_steps):
        X_batch = np.array(sequence2[-n_steps:]).reshape(1, n_steps, 1)
        y_pred = sess.run(outputs, feed_dict={X: X_batch})
        sequence2.append(y_pred[0, -1, 0])

plt.figure(figsize=(11,4))
plt.subplot(121)
plt.plot(t, sequence1, "b-")
plt.plot(t[:n_steps], sequence1[:n_steps], "b-", linewidth=3)
plt.xlabel("Czas")
plt.ylabel("Wartość")

plt.subplot(122)
plt.plot(t, sequence2, "b-")
plt.plot(t[:n_steps], sequence2[:n_steps], "b-", linewidth=3)
plt.xlabel("Czas")
plt.show()
        


