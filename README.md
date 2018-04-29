[![Build Status](https://travis-ci.org/dvgodoy/deepreplay.svg?branch=master)](https://travis-ci.org/dvgodoy/deepreplay)
[![Documentation Status](http://readthedocs.org/projects/deepreplay/badge/?version=latest)](http://deepreplay.readthedocs.io/en/latest/?badge=latest)

# Deep Replay

## Generate visualizations as in my "Hyper-parameters in Action!" series of posts!

***Deep Replay*** is a package designed to allow you to ***replay*** in a visual fashion the training process of a Deep Learning model in Keras, as I have done in my [Hyper-parameter in Action!](https://towardsdatascience.com/hyper-parameters-in-action-a524bf5bf1c) post on [Towards Data Science](http://towardsdatascience.com).

This is an example of what you can do using ***Deep Replay***:

![Part 1 Animation](/images/part1.gif)

It contains:
 - a Keras' callback - ***ReplayData*** - which collects then necessary information, mostly the weights, during the training epochs;
 - a class ***Replay***, which leverages the collected data to build several kinds of visualizations.

The available visualizations are:
 - ***Feature Space***: plot of a 2-D grid representing the twisted and turned feature space,  corresponding to the output of a hidden layer (only 2-unit hidden layers supported for now);
 - ***Probabilities***: histograms of the resulting class probabilities for the inputs, corresponding to the output of the final layer (only binary classification supported for now);
 - ***Loss and Metric***: line plot for the loss and a chosen metric, computed over all the inputs;
 - ***Losses***: histogram of the losses computed over all the inputs (only binary cross-entropy loss suported for now).

Feature Space | Class Probability | Loss/Metric | Losses
:-:|:-:|:-:|:-:
![Feature Space](/images/feature_space.png) | ![Probability Histogram](/images/prob_histogram.png) | ![Loss and Metric](/images/loss_and_metric.png) | ![Loss Histogram](/images/loss_histogram.png)

## Google Colab

Eager to try it out right away? Don't wait any longer!

Open the notebooks directly on Google Colab and try it yourself:

- [Part 1 - Activation Functions](https://colab.research.google.com/github/dvgodoy/deepreplay/blob/master/notebooks/part1_activation_functions.ipynb)

### Installation

To install ***Deep Replay*** from [PyPI](https://pypi.org/project/deepreplay/), just type:
```python
pip install deepreplay
```

### Documentation

You can find the full documentations at [Read the Docs](http://deepreplay.readthedocs.io/).

### Quick Start

To use ***Deep Replay***, you must first create an instance of the Keras' callback, ***ReplayData***, passing as arguments the inputs (X) and outputs (y) you're using to train the model, as well as the filename and group (for more details, see h5py) where you want the collected data to be saved:
```python
from deepreplay.callbacks import ReplayData
from deepreplay.datasets.parabola import load_data

X, y = load_data()

replaydata = ReplayData(X, y, filename='hyperparms_in_action.h5', group_name='part1')
```

Then, create a Keras model of your choice, compile it and fit it, adding the instance of the callback object you just created:
```python
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.initializers import glorot_normal, normal

model = Sequential()
model.add(Dense(input_dim=2,
                units=2,
                activation='sigmoid',
                kernel_initializer=glorot_normal(seed=42),
                name='hidden'))
model.add(Dense(units=1,
                activation='sigmoid',
                kernel_initializer=normal(seed=42),
                name='output'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.05), metrics=['acc'])

model.fit(X, y, epochs=150, batch_size=16, callbacks=[replaydata])
```

After your model finishes training, you'll end up with a HDF5 file (***hyperparms_in_action.h5***, in the example), containing a new group (***part1***, in the example) that holds all the necessary information. The Keras model itself is also automatically saved as ***<group_name>_model.h5***, that is, ***part1_model.h5*** in the example.

Next, it is time to feed the information to a ***Replay*** instance:
```python
from deepreplay.replay import Replay

replay = Replay(replay_filename='hyperparms_in_action.h5', group_name='part1')
```

Then, you can create a regular Matplotlib figure, like:
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1, figsize=(5, 5))
```

And use your ***Replay*** instance to build the visualization of your choice, say, ***Feature Space*** based on the output of the layer named ***hidden***:
```python
fs = replay.build_feature_space(ax, layer_name='hidden')
```

Now, you're ready to make a ***plot*** of your ***Feature Space*** in any given ***epoch***, or to ***animate*** its evolution during the whole training:
```python
fs.plot(epoch=60).savefig('feature_space_epoch60.png', dpi=120)
fs.animate().save('feature_space_animation.mp4', dpi=120, fps=5)
```

The results should look like this:

![Feature Space Epoch 60](/images/feature_space_epoch60.png) ![Feature Space Animation](/images/feature_space_animation.gif)

***TIP***: If you get an error message regarding the ```MovieWriter```, try ```conda install -c conda-forge ffmpeg``` to install FFMPEG, the writer used to generate the animations.

Alternatively, you can explicitly specify a different MovieWriter, for instance, `avconv`:
```python
from matplotlib import animation

Writer = animation.writers['avconv']
metadata = dict(title='Sigmoid Activation Function',
                artist='Hyper-parameters in Action!')
writer = Writer(fps=5, metadata=metadata)

fs.animate().save('feature_space_animation.mp4', dpi=120, writer=writer)
```

### Comments, questions, suggestions, bugs

***DISCLAIMER***: this is a project ***under development***, so it is likely you'll run into bugs/problems.

So, if you find any bugs/problems, please open an [issue](https://github.com/dvgodoy/deepreplay/issues) or submit a [pull request](https://github.com/dvgodoy/deepreplay/pulls).
