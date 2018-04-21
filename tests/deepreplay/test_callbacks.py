import pytest
from keras.utils.test_utils import keras_test

@pytest.fixture
def replay_data():
    import h5py
    replay_data = h5py.File('./tests/rawdata/hyperparms_in_action.h5', 'r')
    yield replay_data['part1_activation_functions']
    replay_data.close()

@pytest.fixture
def model_data():
    from keras.models import load_model
    return load_model('./tests/rawdata/part1_activation_functions_model.h5')

