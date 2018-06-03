import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import glorot_uniform, normal
from deepreplay.datasets.ball import load_data
from deepreplay.callbacks import ReplayData
from deepreplay.replay import Replay
from deepreplay.plot import compose_plots

import matplotlib.pyplot as plt

group_name = 'gradients'

X, y = load_data()

n_layers = 5
variance = 0.01
activation = 'tanh'

init = normal(mean=0, stddev=np.sqrt(variance))
#init = VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform')
#init = glorot_normal() # sigmoid
#init = glorot_uniform() # tanh
#init = he_uniform() # relu
#init_name = 'He Uniform'

model = Sequential()
model.add(Dense(units=100, input_dim=10, activation=activation, kernel_initializer=init, name='h1'))
for i in range(2, n_layers + 1):
    model.add(Dense(units=100, activation=activation, kernel_initializer=init, name='h{}'.format(i)))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init, name='o'))

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['acc'])

replaydata = ReplayData(X, y, filename='gradients.h5', group_name=group_name, model=model)

replay = Replay(replay_filename='gradients.h5', group_name=group_name + '_init')

fig = plt.figure(figsize=(12, 6))
ax_zvalues = plt.subplot2grid((2, 2), (0, 0))
ax_weights = plt.subplot2grid((2, 2), (0, 1))
ax_activations = plt.subplot2grid((2, 2), (1, 0))
ax_gradients = plt.subplot2grid((2, 2), (1, 1))

zv = replay.build_outputs_violins(ax_zvalues, before_activation=True)
wv = replay.build_weights_violins(ax_weights)
av = replay.build_outputs_violins(ax_activations)
gv = replay.build_gradients(ax_gradients)

sample_figure = compose_plots([zv, wv, av, gv], 0)
sample_figure.savefig('gradients.png', dpi=120, format='png')
