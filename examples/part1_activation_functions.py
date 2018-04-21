from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.initializers import glorot_normal, normal

from deepreplay.datasets.parabola import load_data
from deepreplay.callbacks import ReplayData
from deepreplay.replay import Replay
from deepreplay.plot import compose_animations, compose_plots

import matplotlib.pyplot as plt

group_name = 'part1_activation_functions'

X, y = load_data()

sgd = SGD(lr=0.05)

glorot_initializer = glorot_normal(seed=42)
normal_initializer = normal(seed=42)

replaydata = ReplayData(X, y, filename='hyperparms_in_action.h5', group_name=group_name)

model = Sequential()
model.add(Dense(input_dim=2,
                units=2,
                kernel_initializer=glorot_initializer,
                activation='sigmoid',
                name='hidden'))

model.add(Dense(units=1,
                kernel_initializer=normal_initializer,
                activation='sigmoid',
                name='output'))

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['acc'])

model.fit(X, y, epochs=150, batch_size=16, callbacks=[replaydata])

replay = Replay(replay_filename='hyperparms_in_action.h5', group_name=group_name)

fig = plt.figure(figsize=(12, 6))
ax_fs = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
ax_ph_neg = plt.subplot2grid((2, 4), (0, 2))
ax_ph_pos = plt.subplot2grid((2, 4), (1, 2))
ax_lm = plt.subplot2grid((2, 4), (0, 3))
ax_lh = plt.subplot2grid((2, 4), (1, 3))

fs = replay.build_feature_space(ax_fs, layer_name='hidden')
ph = replay.build_probability_histogram(ax_ph_neg, ax_ph_pos)
lh = replay.build_loss_histogram(ax_lh)
lm = replay.build_loss_and_metric(ax_lm, 'acc')

sample_figure = compose_plots([fs, ph, lm, lh], 80)
sample_figure.savefig('part1.png', dpi=120, format='png')

sample_anim = compose_animations([fs, ph, lm, lh])
sample_anim.save(filename='part1.mp4', dpi=120, fps=5)
