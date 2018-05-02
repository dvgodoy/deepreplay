from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.initializers import glorot_normal, normal

from deepreplay.datasets.parabola import load_data
from deepreplay.callbacks import ReplayData
from deepreplay.replay import Replay
from deepreplay.plot import compose_animations, compose_plots

import matplotlib.pyplot as plt

X, y = load_data()

sgd = SGD(lr=0.05)

for activation in ['sigmoid', 'tanh', 'relu']:
    glorot_initializer = glorot_normal(seed=42)
    normal_initializer = normal(seed=42)

    replaydata = ReplayData(X, y, filename='comparison_activation_functions.h5', group_name=activation)

    model = Sequential()
    model.add(Dense(input_dim=2,
                    units=2,
                    kernel_initializer=glorot_initializer,
                    activation=activation,
                    name='hidden'))

    model.add(Dense(units=1,
                    kernel_initializer=normal_initializer,
                    activation='sigmoid',
                    name='output'))

    model.compile(loss='binary_crossentropy',
                  optimizer=sgd,
                  metrics=['acc'])

    model.fit(X, y, epochs=150, batch_size=16, callbacks=[replaydata])

fig, axs = plt.subplots(1, 3, figsize=(12, 4))

replays = []
for activation in ['sigmoid', 'tanh', 'relu']:
    replays.append(Replay(replay_filename='comparison_activation_functions.h5', group_name=activation))

spaces = []
for ax, replay, activation in zip(axs, replays, ['sigmoid', 'tanh', 'relu']):
    space = replay.build_feature_space(ax, layer_name='hidden')
    space.set_title(activation)
    spaces.append(space)

sample_figure = compose_plots(spaces, 80)
sample_figure.savefig('comparison.png', dpi=120, format='png')

#sample_anim = compose_animations(spaces)
#sample_anim.save(filename='comparison.mp4', dpi=120, fps=5)

