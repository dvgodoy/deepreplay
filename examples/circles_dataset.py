from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import SGD
from keras.initializers import normal, he_normal

from deepreplay.callbacks import ReplayData
from deepreplay.replay import Replay
from deepreplay.plot import compose_animations, compose_plots

from sklearn.datasets import make_circles

import matplotlib.pyplot as plt

group_name = 'circles'

X, y = make_circles(n_samples=2000, random_state=27, noise=0.03)

sgd = SGD(lr=0.01)

he_initializer = he_normal(seed=42)
normal_initializer = normal(seed=42)

replaydata = ReplayData(X, y, filename='circles_dataset.h5', group_name=group_name)

model = Sequential()
model.add(Dense(input_dim=2,
                units=5,
                kernel_initializer=he_initializer))
model.add(Activation('relu'))
model.add(Dense(units=3,
                kernel_initializer=he_initializer))
model.add(Activation('relu'))
model.add(Dense(units=2,
                kernel_initializer=normal_initializer,
                activation='linear',
                name='hidden'))
model.add(Dense(units=1,
                kernel_initializer=normal_initializer,
                activation='sigmoid',
                name='output'))

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['acc'])

model.fit(X, y, epochs=300, batch_size=16, callbacks=[replaydata])

replay = Replay(replay_filename='circles_dataset.h5', group_name=group_name)

fig = plt.figure(figsize=(12, 6))
ax_fs = plt.subplot2grid((2, 4), (0, 0), colspan=2, rowspan=2)
ax_ph_neg = plt.subplot2grid((2, 4), (0, 2))
ax_ph_pos = plt.subplot2grid((2, 4), (1, 2))
ax_lm = plt.subplot2grid((2, 4), (0, 3))
ax_lh = plt.subplot2grid((2, 4), (1, 3))

fs = replay.build_feature_space(ax_fs, layer_name='hidden',
                                display_grid=False, scale_fixed=False)
ph = replay.build_probability_histogram(ax_ph_neg, ax_ph_pos)
lh = replay.build_loss_histogram(ax_lh)
lm = replay.build_loss_and_metric(ax_lm, 'acc')

sample_figure = compose_plots([fs, ph, lm, lh], 280)
sample_figure.savefig('circles.png', dpi=120, format='png')

sample_anim = compose_animations([fs, ph, lm, lh])
sample_anim.save(filename='circles.mp4', dpi=120, fps=5)
