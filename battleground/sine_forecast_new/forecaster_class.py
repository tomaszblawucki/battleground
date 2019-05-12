#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


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
                 activation='tanh',
                 forecast_horizon=1,
                 dropout_proba=0.5):
        self.n_steps = n_steps
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.forecast_horizon = forecast_horizon
        self.dropout_proba = dropout_proba
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
        dropout_proba = tf.placeholder(tf.float32, name='dropout_proba')
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
        outputs = tf.reshape(stacked_outputs,
                             [-1, self.n_steps, self.n_outputs],
                             name='outputs')
        loss = tf.reduce_mean(tf.square(outputs-y), name='loss')
        optimizer = self.optimizer(learning_rate=self.learning_rate,
                                   name='optimizer')
        training_op = optimizer.minimize(loss, name='training_op')

    def train(self, X_train, y_train, n_epochs, generator='stochastic'):
        loss_stack, val_loss_stack = {}, {}
        with tf.Session(graph=self.graph) as sess:
            sess.run(self.init_op)
            for epoch in range(n_epochs):
                if generator == 'stochastic':
                    batcher = self.stochastic_sine_generator(self.batch_size, self.n_steps, shift=self.forecast_horizon)
                else:
                    batcher = self.sine_generator(self.batch_size, self.n_steps, shift=self.forecast_horizon)
                for _, X_batch, y_batch in batcher: 
                    feed = {'x_placeholder:0': X_batch,
                            'y_placeholder:0': y_batch,
                            'dropout_proba:0': self.dropout_proba}
#                    sess.run('training_op', feed_dict=feed)
                    loss, _ = sess.run(['loss:0', 'training_op'], feed_dict=feed)
                valid_batcher = self.sine_generator(1000, self.n_steps, shift=self.forecast_horizon, t_min=30, t_max=50)
                _, x_valid, y_valid = next(valid_batcher)
                validation_feed = {'x_placeholder:0': x_valid,
                                   'y_placeholder:0': y_valid,
                                   'dropout_proba:0': self.dropout_proba}
                valid_loss = sess.run('loss:0', feed_dict=validation_feed)
                if not epoch % 50:
                    print(f'EPOCH: {epoch}, LOSS: {loss}, VALIDATION_LOSS: {valid_loss}')
                if not epoch % 10:
                    self.saver.save(sess, "./models/sine_forecaster.ckpt")
                loss_stack[epoch] = loss
                val_loss_stack[epoch] = valid_loss
        return loss_stack, val_loss_stack
    
    def predict(self, X_test):
        with tf.Session(graph=self.graph) as sess:
            self.saver.restore(sess, './models/sine_forecaster.ckpt')
            feed = {'x_placeholder:0': X_test,
                    'dropout_proba:0': 1.0}
            pred = sess.run('outputs:0', feed_dict=feed)
            return pred

    def time_series(self, t):
#        return np.sin(t)
#        return t * np.sin(t)
#        return t * np.sin(t) / 3 + 2 * np.sin(t*5)
        return np.sin(t) / 3 + 2 * np.sin(t*5)

    

    def sine_generator(self, batch_size, seq_len, t_min=0, t_max=30, resolution=0.1, shift=1):
        shift *= resolution
        rng = t_max - t_min
        t = np.linspace(t_min, t_max, int(rng//resolution) + 1)
        t_len = t.shape[0]
        n_batches = int(np.ceil(t_len / (seq_len * batch_size)))
        r = t_len % seq_len
        if r > 0:
            t = t[:-r]
        for batch_idx in range(n_batches):
            res = t[batch_idx * seq_len * batch_size : (batch_idx+1) * seq_len * batch_size]
            res = res.reshape(-1, seq_len, 1)
            yield res, self.time_series(res), self.time_series(res + shift)
        return

    def stochastic_sine_generator(self, batch_size, seq_len, t_min=0, t_max=30, resolution=0.1, shift=1, random_density=1):
        shift *= resolution
        rng = (t_max - t_min)
        n_ticks = int(rng/resolution)
        if n_ticks < seq_len:
            raise ValueError('sequence length cannot be greater than generator range')
        t0 = np.random.uniform(t_min, t_max-(seq_len*resolution), rng * random_density)
        t = np.arange(0, seq_len)*resolution
        tr = t + t0.reshape(-1, 1)
        tr_indexes = list(range(tr.shape[0]))
        for _t in tr:
            _t = np.random.choice(tr_indexes, batch_size)
            _t = tr[_t].reshape(-1, seq_len, 1)
            yield _t, self.time_series(_t), self.time_series(_t + shift) 
        return
