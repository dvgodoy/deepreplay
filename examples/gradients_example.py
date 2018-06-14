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
zv = replay.build_outputs(ax_zvalues, before_activation=True)
wv = replay.build_weights(ax_weights)
av = replay.build_outputs(ax_activations)
gv = replay.build_gradients(ax_gradients)

sample_figure = compose_plots([zv, wv, av, gv], 0, title='Activation: {} - Initializer: {}'.format(activation, init_title))
sample_figure.savefig('gradients.png', dpi=120, format='png')


# Showdown
input_dim = 10

results = {}
pairs = [('tanh', 'glorot_normal'),
         ('tanh', 'glorot_uniform'),
         ('relu', 'he_normal'),
         ('relu', 'he_uniform')]

X, y = load_data(n_dims=input_dim)
y0 = np.zeros_like(y)
y1 = np.ones_like(y)
yr = np.random.randint(2, size=y.shape)

ys = [y, y0, y1, yr]

for y, yname in zip(ys, ['y', 'y0', 'y1', 'yr']):
    results.update({yname: {}})
    for n_layers in [5, 20, 50, 100]:
        results[yname].update({n_layers: {}})
        units = [100] * n_layers
        for pair in pairs:
            activation, init_key = pair
            group_name = '{}_{}_{}_{}'.format(activation, init_key, n_layers, yname)
            key_name = '{}_{}'.format(activation, init_key)
            print(group_name)

            init = initializers[init_key]['init']

            model = build_model(n_layers,
                                input_dim=input_dim,
                                units=units,
                                activation=activation,
                                initializer=init)

            replaydata = ReplayData(X, y, filename='showdown.h5', group_name=group_name, model=model)
            replay = Replay(replay_filename='showdown.h5', group_name=group_name)
            replay.build_weights(None)
            replay.build_gradients(None)

            gstd = replay.gradients_std[0][:-1]
            wstd = replay.weights_std[0][:-1]
            results[yname][n_layers].update({key_name: {'gradient': gstd, 'weights': wstd, 'ratio': gstd / wstd}})