#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 09:24:36 2019

@author: tom
"""
import pickle
import matplotlib.pyplot as plt
from forecaster_class import SineForecaster
import os

def run(BATCH_SIZE = 1,
        N_STEPS = 20,
        FORECAST_HORIZON = 20,
        N_NEURONS = 125,
        N_LAYERS = 1,
        DROPOUT_PROBA = 1.0,
        GENERATOR = 'stochastic',
        N_EPOCHS = 200,
        RNN_CELL = 'LSTMCell',
        ACTIVATION = 'tanh',
        OPTIMIZER = 'AdamOptimizer',
        PARAM_MODE = None):
    
    sf = SineForecaster(n_steps=N_STEPS,
                        n_inputs=1,
                        n_outputs=1,
                        n_neurons=N_NEURONS,
                        n_layers=N_LAYERS,
                        learning_rate=0.001,
                        rnn_cell=RNN_CELL,
                        batch_size=BATCH_SIZE,
                        optimizer=OPTIMIZER,
                        activation=ACTIVATION,
                        forecast_horizon=FORECAST_HORIZON,
                        dropout_proba=DROPOUT_PROBA)
    
    train_loss, valid_loss = sf.train(None, None, N_EPOCHS, generator=GENERATOR)
    
    gen = sf.sine_generator(4, N_STEPS, t_min=30, t_max=50, shift=FORECAST_HORIZON)
    t_test, x_test, y_test = next(gen)
    pred = sf.predict(x_test)
    plt.figure(figsize=(15, 8))
    plt.plot(t_test.ravel() - FORECAST_HORIZON * 0.1, x_test.ravel(), 'b', linewidth=5, alpha=0.2)
    plt.plot(t_test.ravel(), y_test.ravel(), 'g', linewidth=3, alpha=0.6)
    plt.plot(t_test.ravel(), pred.ravel(), 'r--')
    plt.grid()
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_loss.values())
    plt.plot(valid_loss.values())
    
    if PARAM_MODE:
        params = {
                'BATCH_SIZE': BATCH_SIZE,
                'N_STEPS': N_STEPS,
                'FORECAST_HORIZON': FORECAST_HORIZON,
                'N_NEURONS': N_NEURONS,
                'N_LAYERS': N_LAYERS,
                'DROPOUT_PROBA': DROPOUT_PROBA,
                'GENERATOR': GENERATOR,
                'N_EPOCHS': N_EPOCHS,
                'RNN_CELL': RNN_CELL,
                'ACTIVATION': ACTIVATION,
                'OPTIMIZER':OPTIMIZER,
                }
        
        
        train_file_name = f'train_stat__batchSz_{BATCH_SIZE}#nSteps_{N_STEPS}#nNeurons_{N_NEURONS}#nLayers_{N_LAYERS}#dropout_{DROPOUT_PROBA}#generator_{GENERATOR}#cell_{RNN_CELL}#activation_{ACTIVATION}#optimizer_{OPTIMIZER}'
        valid_file_name = f'valid_stat__batchSz_{BATCH_SIZE}#nSteps_{N_STEPS}#nNeurons_{N_NEURONS}#nLayers_{N_LAYERS}#dropout_{DROPOUT_PROBA}#generator_{GENERATOR}#cell_{RNN_CELL}#activation_{ACTIVATION}#optimizer_{OPTIMIZER}'
            
        f = open(f"./stats/{PARAM_MODE}/{valid_file_name}.pkl", "wb")
        pickle.dump(valid_loss, f)
        pickle.dump(params, f)
        f.close()
        
        f = open(f"./stats/{PARAM_MODE}/{train_file_name}.pkl", "wb")
        pickle.dump(train_loss, f)
        pickle.dump(params, f)
        f.close()
    
    return sf

# =============================================================================
# Check batch size hyperparameter influence
# =============================================================================

PARAM_MODE = 'OPTIMIZER'


def create_dir(path):
    if os.path.exists(path):
        for i in os.listdir(path):
            os.remove(os.path.join(path, i))
        os.rmdir(path)
    os.mkdir(path)
    
create_dir(f'./stats/{PARAM_MODE}')
for PARAM in ['AdamOptimizer', 'GradientDescentOptimizer', 'AdagradOptimizer', 'RMSPropOptimizer']:
    print(f'PARAM:{PARAM_MODE} => {PARAM}')
    run(BATCH_SIZE=5,
        N_STEPS=100,
        N_EPOCHS=600,
        N_NEURONS=125,
        N_LAYERS=1,
        DROPOUT_PROBA=0.75,
        OPTIMIZER='GradientDescentOptimizer',
        PARAM_MODE=PARAM_MODE)



#sf = run(BATCH_SIZE=100)

files = os.listdir(f'./stats/{PARAM_MODE}')    


plt.figure(figsize=(15,8))
#plt.yscale('log')
for file_name in files:
    if file_name[:5] == 'valid':
        f = open(f'./stats/{PARAM_MODE}/{file_name}', 'rb')
        stat=pickle.load(f)
        params=pickle.load(f)
        plt.plot(stat.values(), label=f'{PARAM_MODE}: {params[PARAM_MODE]}')
        plt.legend()
        f.close()
plt.title('validation_loss')


plt.figure(figsize=(15,8))
plt.yscale('log')
for file_name in files:
    if file_name[:5] == 'train':
        f = open(f'./stats/{PARAM_MODE}/{file_name}', 'rb')
        stat=pickle.load(f)
        params=pickle.load(f)
        plt.plot(stat.values(), label=f'{PARAM_MODE}: {params[PARAM_MODE]}')
        plt.legend()
        f.close()
plt.title('train_loss')

# =============================================================================
# MODEL TEST
# =============================================================================

sf = SineForecaster(n_steps=100,
                    n_inputs=1,
                    n_outputs=1,
                    n_neurons=125,
                    n_layers=1,
                    learning_rate=0.001,
                    rnn_cell='LSTMCell',
                    batch_size=5,
                    optimizer='GradientDescentOptimizer',
                    activation='tanh',
                    forecast_horizon=20,
                    dropout_proba=0.75)

sf.train(None,None, n_epochs=200)

plt.figure(figsize=(12, 6))
test_gen = sf.sine_generator(1000, 100, 80, 100, shift=20)
ticks, test_x, test_y = next(test_gen)

pred_vals = sf.predict(test_x)

plt.plot(ticks.ravel(), test_y.ravel(), linewidth=3)
plt.plot(ticks.ravel(), pred_vals.ravel())


#stochastic_gen = sf.sine_generator(1, 100, shift=20)
#
#for t, x, y in stochastic_gen:
#    plt.plot(t.ravel(), x.ravel())
#    plt.plot(t.ravel(), y.ravel())

