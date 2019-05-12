#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

class SineForecaster():
    def __init__(self,
                 n_steps,
                 n_inputs,
                 n_outputs,
                 n_neurons=100,
                 n_layers=1,
                 learning_rate=0.001,
                 rnn_cell='LSTMCell',
                 batch_size=1,
                 optimizer='AdamOptimizer',
                 activation='tanh'):
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.activation = getattr(tf.nn, activation)
        self.rnn_cell = getattr(tf.nn.rnn_cell, rnn_cell)
        self.optimizer = getattr(tf.train, optimizer)
        self.graph = tf.Graph()
        with self.graph.as_default():  
            tf.set_random_seed(42)
            self.build()
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
            
    def build(self):
        x = tf.placeholder(tf.float32,
                           shape=(None, self.n_steps, self.n_inputs),
                           name='x_placeholder')
        y = tf.placeholder(tf.float32,
                           shape=(None, self.n_steps, self.n_outputs),
                           name='y_placeholder')
        dropout_proba =tf.placeholder(tf.float32, name='dropout_proba')
        cells = tf.nn.rnn_cell.MultiRNNCell([
                tf.nn.rnn_cell.DropoutWrapper(
                        self.rnn_cell(
                                    num_units=self.n_neurons,
                                    activation=self.activation), 
                                output_keep_prob=dropout_proba)
                 for _ in range(self.n_layers)])
        lstm_outputs, state = tf.nn.dynamic_rnn(cells, x, dtype=tf.float32)
        stacked_lstm_outputs = tf.reshape(lstm_outputs, [-1, self.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_lstm_outputs, self.n_outputs)
        outputs = tf.reshape(stacked_outputs, [-1, self.n_steps, self.n_outputs], name='outputs')
        loss = tf.reduce_mean(tf.square(outputs-y), name='loss')
        optimizer = self.optimizer(learning_rate=self.learning_rate, name='optimizer')
        training_op = optimizer.minimize(loss, name='training_op')
        
    def train(self, X_train, y_train, n_epochs):
        with tf.Session(graph=self.graph) as sess:
            sess.run(self.init_op)
            for epoch in range(n_epochs):
                X_batch, y_batch = self.next_batch(batch_size=self.batch_size, n_steps=self.n_steps)
                feed = {'x_placeholder:0':X_batch,
                        'y_placeholder:0':y_batch,
                        'dropout_proba:0':0.5}
                loss, _ = sess.run(['loss:0', 'training_op'], feed_dict=feed)
                if not epoch % 50:
                    print(f'EPOCH: {epoch}, LOSS: {loss}')
                if not epoch % 10:
                    saver.save(sess, "./models/sine_forecaster.ckpt")
   
                
    def next_batch(self, batch_size, n_steps):
        t0 = np.arange(0., 1., 1/batch_size)*(t_max - t_min - n_steps * resolution)
        t0 = t0.reshape(-1, 1)
        Ts = t0 + np.arange(0., n_steps + 1) * resolution
        ys = Ts * np.sin(Ts) / 3 + 2 * np.sin(Ts*5)
        return ys[:, :-1].reshape(-1, n_steps, 1), ys[:, 1:].reshape(-1, n_steps, 1)
    
    def predict(self, X_test):
        with tf.Session(graph=self.graph) as sess:
            saver.restore(sess, './models/sine_forecaster.ckpt')
            feed = {'x_placeholder:0':X_test,
                    'dropout_proba:0':1.0}
            pred = sess.run('outputs:0', feed_dict=feed)
            return pred
            
    
    
sf = SineForecaster(20, 1, 1)
#sf.train(None, None, 1000)
x_test, y_test = sf.next_batch(1, 20)
pred = sf.predict(x_test)
print(pred)

import matplotlib.pyplot as plt
plt.plot(pred.ravel())
#plt.plot(x_test.ravel())
plt.plot(y_test.ravel())