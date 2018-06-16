import os
import pytest
import numpy as np
import numpy.testing as npt

FIXTURE_DIR = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'rawdata')

@pytest.fixture(scope='module')
def replay_data():
    import h5py
    replay_data = h5py.File(os.path.join(FIXTURE_DIR, 'hyperparms_in_action.h5'), 'r')
    return replay_data['part1_activation_functions']

@pytest.fixture(scope='module')
def model_data():
    from keras.models import load_model
    return load_model(os.path.join(FIXTURE_DIR, 'part1_activation_functions_model.h5'))

@pytest.fixture(scope='module')
def training_data(tmpdir_factory):
    import h5py
    from keras.layers import Dense
    from keras.models import Sequential
    from keras.optimizers import SGD
    from keras.initializers import glorot_normal, normal

    from deepreplay.datasets.parabola import load_data
    from deepreplay.callbacks import ReplayData

    filename = str(tmpdir_factory.mktemp('data').join('training.h5'))

    X, y = load_data(xlim=(-1, 1), n_points=1000, shuffle=True, seed=13)

    sgd = SGD(lr=0.05)

    glorot_initializer = glorot_normal(seed=42)
    normal_initializer = normal(seed=42)

    replaydata = ReplayData(X, y, filename=filename, group_name='part1_activation_functions')

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

    model.fit(X, y, epochs=20, batch_size=16, callbacks=[replaydata])

    training_data = h5py.File(filename, 'r')
    return training_data['part1_activation_functions']

def test_dataset(replay_data, training_data):
    npt.assert_allclose(replay_data['inputs'], training_data['inputs'], atol=1e-5)
    npt.assert_allclose(replay_data['targets'], training_data['targets'], atol=1e-5)

def test_attrs(replay_data, training_data):
    for attr in ['samples', 'batch_size', 'n_batches', 'n_epochs', 'n_layers', 'activation_functions']:
        npt.assert_equal(replay_data.attrs[attr], training_data.attrs[attr])

def test_weights(replay_data, training_data):
    layers = filter(lambda key: 'layer' in key, replay_data.keys())
    for layer in layers:
        for weight in replay_data[layer].keys():
            print(replay_data[layer][weight])
            print(training_data[layer][weight])
            npt.assert_allclose(replay_data[layer][weight], training_data[layer][weight], atol=1e-3)

def test_metrics(replay_data, training_data, model_data):
    npt.assert_allclose(replay_data['loss'], training_data['loss'], atol=1e-3)
    for metric in model_data.metrics:
        npt.assert_allclose(replay_data[metric], training_data[metric], atol=1e-3)