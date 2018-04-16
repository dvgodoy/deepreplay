from __future__ import division
import numpy as np
import h5py
import keras.backend as K
from keras.models import load_model
from plot import build_2d_grid, FeatureSpace, ProbabilityHistogram, LossHistogram, LossAndMetric
from plot import FeatureSpaceData, FeatureSpaceLines, ProbHistogramData, LossHistogramData, LossAndMetricData

TRAINING_MODE = 1
TEST_MODE = 0

class Replay(object):
    def __init__(self, replay_filename, group_name):
        self.learning_phase = TEST_MODE

        self.model = load_model('{}_model.h5'.format(group_name))
        self.replay_data = h5py.File('{}'.format(replay_filename), 'r')
        self.group_name = group_name
        self.group = self.replay_data[self.group_name]
        self.inputs = self.group['inputs'][:]
        self.targets = self.group['targets'][:]
        self.n_epochs = self.group.attrs['n_epochs']
        self.n_layers = self.group.attrs['n_layers']
        self.weights = self._retrieve_weights()
        self._get_output = K.function(inputs=[K.learning_phase()] + self.model.inputs + self.model.weights,
                                      outputs=[self.model.layers[-1].output])
        self._get_metrics = K.function(inputs=[K.learning_phase()] + self.model.inputs + self.model.targets +
                                              self.model.weights + self.model.sample_weights,
                                       outputs=[self.model.total_loss] + self.model.metrics_tensors)
        self._get_binary_crossentropy = K.function(inputs=[K.learning_phase()] + self.model.inputs + self.model.targets +
                                                          self.model.weights + self.model.sample_weights,
                                                   outputs=[K.binary_crossentropy(self.model.targets[0],
                                                                                  self.model.outputs[0])])
        self.feature_space_data = None
        self.loss_hist_data = None
        self.loss_and_metric_data = None
        self.prob_hist_data = None

        self.feature_space_plot = None
        self.loss_hist_plot = None
        self.loss_and_metric_plot = None
        self.prob_hist_plot = None

    def _retrieve_weights(self):
        n_weights = [range(len(self.group['layer{}'.format(l)]))
                     for l in range(self.n_layers)]
        weights = [np.array(self.group['layer{}'.format(l)]['weights{}'.format(w)])
                   for l, ws in enumerate(n_weights)
                   for w in ws]
        return [[w[epoch] for w in weights] for epoch in range(self.n_epochs)]

    @property
    def feature_space(self):
        return self.feature_space_plot, self.feature_space_data

    @property
    def loss_histogram(self):
        return self.loss_hist_plot, self.loss_hist_data

    @property
    def loss_and_metric(self):
        return self.loss_and_metric_plot, self.loss_and_metric_data

    @property
    def probability_histogram(self):
        return self.prob_hist_plot, self.prob_hist_data

    @property
    def training_loss(self):
        return self.group['loss'][:]

    def get_training_metric(self, metric_name):
        try:
            metric = self.group[metric_name][:]
        except KeyError:
            metric = np.zeros(shape=(self.n_epochs,))
        return metric

    def _make_function(self, layer):
        return K.function(inputs=self.model.inputs + self.model.weights,
                          outputs=[layer.output])

    def _predict_proba(self, inputs, weights):
        return self._get_output([self.learning_phase, inputs] + weights)

    def predict_proba(self, epoch_start=1, epoch_end=-1):
        epoch_start -= 1
        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        probas = []
        for epoch in range(epoch_start, epoch_end):
            weights = self.weights[epoch]
            probas.append(self._predict_proba(self.inputs, weights)[0])
        probas = np.array(probas)
        return probas

    def build_loss_histogram(self, ax, epoch_start=1, epoch_end=-1):
        if self.model.loss != 'binary_crossentropy':
            raise NotImplementedError("Only binary cross-entropy is supported!")

        epoch_start -= 1
        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)
        binary_xentropy = []
        for epoch in range(epoch_start, epoch_end):
            weights = self.weights[epoch]

            inputs = [self.learning_phase, self.inputs, self.targets] + weights + [np.ones(shape=self.inputs.shape[0])]
            binary_xentropy.append(self._get_binary_crossentropy(inputs=inputs)[0].squeeze())

        binary_xentropy = np.array(binary_xentropy)

        self.loss_hist_data = LossHistogramData(loss=binary_xentropy)
        self.loss_hist_plot = LossHistogram(ax).load_data(self.loss_hist_data)
        return self.loss_hist_plot

    def build_loss_and_metric(self, ax, metric_name, epoch_start=1, epoch_end=-1):
        epoch_start -= 1
        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        evaluations = []
        for epoch in range(epoch_start, epoch_end):
            weights = self.weights[epoch]

            inputs = [self.learning_phase, self.inputs, self.targets] + weights + [np.ones(shape=self.inputs.shape[0])]
            evaluations.append(self._get_metrics(inputs=inputs))

        evaluations = np.array(evaluations)
        loss = evaluations[:, 0]
        try:
            metric = evaluations[:, self.model.metrics_names.index(metric_name)]
        except ValueError:
            metric = np.zeros(shape=(epoch_end - epoch_start,))

        self.loss_and_metric_data = LossAndMetricData(loss=loss, metric=metric, metric_name=metric_name)
        self.loss_and_metric_plot = LossAndMetric(ax).load_data(self.loss_and_metric_data)
        return self.loss_and_metric_plot

    def build_probability_histogram(self, ax_negative, ax_positive, epoch_start=1, epoch_end=-1):
        epoch_start -= 1
        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        self.prob_hist_data = ProbHistogramData(prob=self.predict_proba(epoch_start, epoch_end), target=self.targets)
        self.prob_hist_plot = ProbabilityHistogram(ax_negative, ax_positive).load_data(self.prob_hist_data)
        return self.prob_hist_plot

    def build_feature_space(self, ax, layer_name, grid_points= 1000, epoch_start=1, epoch_end=-1):
        epoch_start -= 1
        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        X = self.inputs
        y = self.targets
        y_ind = y.squeeze().argsort()
        X = X.squeeze()[y_ind].reshape(X.shape)

        xlim = [-1, 1]
        ylim = [-1, 1]

        grid_lines = build_2d_grid(xlim, ylim)
        contour_lines = build_2d_grid(xlim, ylim, grid_points, grid_points)

        layer = self.model.get_layer(layer_name)
        get_activations = self._make_function(layer)
        get_predictions = self._make_function(self.model.layers[-1])

        bent_lines = []
        bent_inputs = []
        bent_contour_lines = []
        bent_preds = []

        for epoch in range(epoch_start, epoch_end):
            weights = self.weights[epoch]

            inputs = [grid_lines.reshape(-1, 2)] + weights
            output_shape = (grid_lines.shape[:2]) + (-1,)
            bent_lines.append(get_activations(inputs=inputs)[0].reshape(output_shape))

            inputs = [X] + weights
            bent_inputs.append(get_activations(inputs=inputs)[0])

            inputs = [contour_lines.reshape(-1, 2)] + weights
            output_shape = (contour_lines.shape[:2]) + (-1,)
            bent_contour_lines.append(get_activations(inputs=inputs)[0].reshape(output_shape))
            bent_preds.append((get_predictions(inputs=inputs)[0].reshape(output_shape) > .5).astype(np.int))

        bent_lines = np.array(bent_lines)
        bent_inputs = np.array(bent_inputs)
        bent_contour_lines = np.array(bent_contour_lines)
        bent_preds = np.array(bent_preds)

        line_data = FeatureSpaceLines(grid=grid_lines, input=X, contour=contour_lines)
        bent_line_data = FeatureSpaceLines(grid=bent_lines, input=bent_inputs, contour=bent_contour_lines)
        self.feature_space_data = FeatureSpaceData(line=line_data, bent_line=bent_line_data, prediction=bent_preds)
        self.feature_space_plot =  FeatureSpace(ax).load_data(self.feature_space_data)
        return self.feature_space_plot
