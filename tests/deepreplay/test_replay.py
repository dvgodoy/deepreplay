import os
import pytest
import numpy as np
import numpy.testing as npt

import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

FIXTURE_DIR = os.path.join(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0], 'rawdata')

@pytest.fixture(scope='module')
def replay():
    """Loads training data into Replay object.
    """
    from deepreplay.replay import Replay
    return Replay(replay_filename=os.path.join(FIXTURE_DIR, 'hyperparms_in_action.h5'),
                  group_name='part1_activation_functions',
                  model_filename=os.path.join(FIXTURE_DIR, 'part1_activation_functions_model.h5'))

@pytest.fixture(scope='module')
def epoch_models():
    """Loads 20 checkpointed Keras' models, one for each trained epoch
    These models will provide the expected values to be matched against computed values in the Replay object.
    """
    from keras.models import load_model
    models = [load_model(os.path.join(FIXTURE_DIR, 'part1_activation_functions_epoch{:02}.h5'.format(i + 1)))
              for i in range(20)]
    for model in models:
        model._make_predict_function()
    return models

def test_predict_proba(replay, epoch_models):
    actual = replay.predict_proba()[1:]
    expected = np.array([model.predict_proba(replay.inputs) for model in epoch_models])
    npt.assert_allclose(actual, expected, atol=1e-6)

def test_build_probability_histogram(replay, epoch_models):
    _, (ax1, ax2) = plt.subplots(1, 2)
    replay.build_probability_histogram(ax1, ax2, epoch_start=2, epoch_end=19)
    _, data = replay.probability_histogram

    actual_prob = data.prob
    expected_prob = np.array([model.predict_proba(replay.inputs) for model in epoch_models[1:-1]])

    npt.assert_allclose(actual_prob, expected_prob, atol=1e-5)

def test_build_loss_and_metric(replay, epoch_models):
    _, ax = plt.subplots(1, 1)
    replay.build_loss_and_metric(ax, 'acc', epoch_start=2, epoch_end=19)
    _, data = replay.loss_and_metric

    actual_loss = data.loss
    actual_acc = data.metric

    expected_loss = []
    expected_acc = []
    for model in epoch_models[1:-1]:
        loss, acc = model.evaluate(replay.inputs, replay.targets)
        expected_loss.append(loss)
        expected_acc.append(acc)

    npt.assert_allclose(actual_loss, expected_loss, atol=1e-5)
    npt.assert_allclose(actual_acc, expected_acc, atol=1e-5)

def test_build_loss_histogram(replay, epoch_models):
    _, ax = plt.subplots(1, 1)
    replay.build_loss_histogram(ax, epoch_start=2, epoch_end=19)
    _, data = replay.loss_histogram

    actual_loss = data.loss[:, :10]

    expected_loss = []
    for model in epoch_models[1:-1]:
        expected_loss.append([])
        for i in range(10):
            loss, _ = model.evaluate(replay.inputs[i, np.newaxis], replay.targets[i, np.newaxis])
            expected_loss[-1].append(loss)

    npt.assert_allclose(actual_loss, expected_loss, atol=1e-5)

def test_build_feature_space(replay, epoch_models):
    from deepreplay.plot import build_2d_grid
    from keras.models import Model

    contour_points = 30
    _, ax = plt.subplots(1, 1)
    replay.build_feature_space(ax, 'hidden', contour_points=contour_points, epoch_start=2, epoch_end=19)
    _, data = replay.feature_space
    _, bent_line, actual_prediction = data
    _, _, actual_bent_contour = bent_line

    contour_lines = build_2d_grid((-1, 1), (-1, 1), contour_points, contour_points).reshape(-1, 2)

    expected_prediction = []
    expected_bent_contour = []
    for model in epoch_models[1:-1]:
        expected_prediction.append(model.predict_classes(contour_lines))
        activations = Model(inputs=model.input,
                            outputs=model.get_layer('hidden').output)
        expected_bent_contour.append(activations.predict(contour_lines))

    expected_prediction = np.array(expected_prediction).reshape(18, contour_points, contour_points)
    expected_bent_contour = np.array(expected_bent_contour).reshape(18, contour_points, contour_points, 2)

    npt.assert_allclose(actual_prediction.squeeze(), expected_prediction, atol=1e-5)
    npt.assert_allclose(actual_bent_contour.squeeze(), expected_bent_contour, atol=1e-5)
