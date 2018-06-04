import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import glorot_uniform, normal, he_uniform
from deepreplay.datasets.ball import load_data
from deepreplay.callbacks import ReplayData
from deepreplay.replay import Replay
from deepreplay.plot import compose_plots

import matplotlib.pyplot as plt

group_name = 'gradients'

X, y = load_data(n_dims=10)

n_layers = 5
variance = 0.01
activation = 'tanh'

init = normal(mean=0, stddev=np.sqrt(variance))
init_name = 'Normal'
init_parms = r'$\sigma = {}$'.format(np.sqrt(variance))
init_title = '{} {}'.format(init_name, init_parms)
#init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')
#init = glorot_normal() # sigmoid
#init = glorot_uniform() # tanh
#init = he_uniform() # relu
#init_name = 'He Uniform'

def build_model(n_layers, input_dim, units, activation, initializer):
    model = Sequential()
    model.add(Dense(units=units, input_dim=input_dim, activation=activation, kernel_initializer=initializer, name='h1'))
    for i in range(2, n_layers + 1):
        model.add(Dense(units=units, activation=activation, kernel_initializer=initializer, name='h{}'.format(i)))
    model.add(Dense(units=1, activation='sigmoid', kernel_initializer=initializer, name='o'))
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])
    return model

model = build_model(n_layers, 10, 100, activation, init)

replaydata = ReplayData(X, y, filename='gradients.h5', group_name=group_name, model=model)

replay = Replay(replay_filename='gradients.h5', group_name=group_name)

fig = plt.figure(figsize=(12, 6))
ax_zvalues = plt.subplot2grid((2, 2), (0, 0))
ax_weights = plt.subplot2grid((2, 2), (0, 1))
ax_activations = plt.subplot2grid((2, 2), (1, 0))
ax_gradients = plt.subplot2grid((2, 2), (1, 1))

layers = ['h1', 'h2', 'h3', 'h4', 'h5', 'o']
zv = replay.build_outputs_violins(ax_zvalues, before_activation=True)
wv = replay.build_weights_violins(ax_weights)
av = replay.build_outputs_violins(ax_activations)
gv = replay.build_gradients(ax_gradients)

sample_figure = compose_plots([zv, wv, av, gv], 0, title='Activation: {} - Initializer: {}'.format(activation, init_title))
sample_figure.savefig('gradients.png', dpi=120, format='png')
