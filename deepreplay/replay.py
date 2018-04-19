from __future__ import division
import numpy as np
import h5py
import keras.backend as K
from keras.models import load_model
from .plot import build_2d_grid, FeatureSpace, ProbabilityHistogram, LossHistogram, LossAndMetric
from .plot import FeatureSpaceData, FeatureSpaceLines, ProbHistogramData, LossHistogramData, LossAndMetricData

TRAINING_MODE = 1
TEST_MODE = 0

class Replay(object):
    """Creates an instance of Replay, to process information collected
    by the callback and generate data to feed the supported visualiza-
    tions.

    Parameters
    ----------
    replay_filename: String
        HDF5 filename used by the callback to store the training
        information.
    group_name: String
        Group inside the HDF5 file where the information was saved.
    model_filename: String, optional
        HDF5 filename of the saved Keras model. Default is the
        group_name with '_model' appended to it.

    Attributes
    ----------
    feature_space: (FeatureSpace, FeatureSpaceData)
        FeatureSpace object to be used for plotting and animating;
        namedtuple containing information about original grid lines,
         data points and predictions.

    loss_histogram: (LossHistogram, LossHistogramData)
        LossHistogram object to be used for plotting and animating;
        namedtuple containing information about example's losses.

    loss_and_metric: (LossAndMetric, LossAndMetricData)
        LossAndMetric object to be used for plotting and animating;
        namedtuple containing information about loss and a given
        metric.

    probability_histogram: (ProbabilityHistogram, ProbHistogramData)
        ProbabilityHistogram object to be used for plotting and
        animating; namedtuple containing information about
        classification probabilities and targets.

    training_loss: ndarray
        An array of shape (n_epochs, ) with training loss as reported
        by Keras at the end of each epoch.
    """
    def __init__(self, replay_filename, group_name, model_filename=''):
        self.learning_phase = TEST_MODE
        if model_filename == '':
            model_filename = '{}_model.h5'.format(group_name)
        self.model = load_model(model_filename)
        self.replay_data = h5py.File('{}'.format(replay_filename), 'r')
        self.group_name = group_name
        self.group = self.replay_data[self.group_name]
        self.inputs = self.group['inputs'][:]
        self.targets = self.group['targets'][:]
        self.n_epochs = self.group.attrs['n_epochs']
        self.n_layers = self.group.attrs['n_layers']
        self.weights = self._retrieve_weights()

        self._model_weights = [w for layer in self.model.layers for w in layer.weights]
        self._get_output = K.function(inputs=[K.learning_phase()] + self.model.inputs + self._model_weights,
                                      outputs=[self.model.layers[-1].output])
        self._get_metrics = K.function(inputs=[K.learning_phase()] + self.model.inputs + self.model.targets +
                                              self._model_weights + self.model.sample_weights,
                                       outputs=[self.model.total_loss] + self.model.metrics_tensors)
        self._get_binary_crossentropy = K.function(inputs=[K.learning_phase()] + self.model.inputs +
                                                          self.model.targets + self._model_weights +
                                                          self.model.sample_weights,
                                                   outputs=[K.binary_crossentropy(self.model.targets[0],
                                                                                  self.model.outputs[0])])
        self._feature_space_data = None
        self._loss_hist_data = None
        self._loss_and_metric_data = None
        self._prob_hist_data = None

        self._feature_space_plot = None
        self._loss_hist_plot = None
        self._loss_and_metric_plot = None
        self._prob_hist_plot = None

    def _retrieve_weights(self):
        n_weights = [range(len(self.group['layer{}'.format(l)]))
                     for l in range(self.n_layers)]
        weights = [np.array(self.group['layer{}'.format(l)]['weights{}'.format(w)])
                   for l, ws in enumerate(n_weights)
                   for w in ws]
        return [[w[epoch] for w in weights] for epoch in range(self.n_epochs)]

    def _make_function(self, layer):
        return K.function(inputs=self.model.inputs + self._model_weights,
                          outputs=[layer.output])

    def _predict_proba(self, inputs, weights):
        return self._get_output([self.learning_phase, inputs] + weights)

    @property
    def feature_space(self):
        return self._feature_space_plot, self._feature_space_data

    @property
    def loss_histogram(self):
        return self._loss_hist_plot, self._loss_hist_data

    @property
    def loss_and_metric(self):
        return self._loss_and_metric_plot, self._loss_and_metric_data

    @property
    def probability_histogram(self):
        return self._prob_hist_plot, self._prob_hist_data

    @property
    def training_loss(self):
        return self.group['loss'][:]

    def get_training_metric(self, metric_name):
        """

        Parameters
        ----------
        metric_name: String
            Metric to return values for.

        Returns
        -------
        metric: ndarray
            An array of shape (n_epochs, ) with the metric as reported
            by Keras at the end of each epoch.
            If the metric was not computed, returns an array of zeros
            with the same shape.
        """
        try:
            metric = self.group[metric_name][:]
        except KeyError:
            metric = np.zeros(shape=(self.n_epochs,))
        return metric

    def predict_proba(self, epoch_start=1, epoch_end=-1):
        """

        Parameters
        ----------
        epoch_start: int, optional
            Initial epoch to return predicted probabilities for.
        epoch_end: int, optional
            Final epoch to return predicted probabilities for.

        Returns
        -------
        probas: ndarray
            An array of shape (n_epochs, n_samples, 2)
        """
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
        """

        Parameters
        ----------
        ax: AxesSubplot

        epoch_start: int, optional

        epoch_end: int, optional

        Returns
        -------
        loss_hist_plot: LossHistogram
        """
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

        self._loss_hist_data = LossHistogramData(loss=binary_xentropy)
        self._loss_hist_plot = LossHistogram(ax).load_data(self._loss_hist_data)
        return self._loss_hist_plot

    def build_loss_and_metric(self, ax, metric_name, epoch_start=1, epoch_end=-1):
        """

        Parameters
        ----------
        ax: AxesSubplot

        metric_name: String

        epoch_start: int, optional

        epoch_end: int, optional

        Returns
        -------
        loss_and_metric_plot: LossAndMetric
        """
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

        self._loss_and_metric_data = LossAndMetricData(loss=loss, metric=metric, metric_name=metric_name)
        self._loss_and_metric_plot = LossAndMetric(ax).load_data(self._loss_and_metric_data)
        return self._loss_and_metric_plot

    def build_probability_histogram(self, ax_negative, ax_positive, epoch_start=1, epoch_end=-1):
        """

        Parameters
        ----------
        ax_negative: AxesSubplot

        ax_positive: AxesSubplot

        epoch_start: int, optional

        epoch_end: int, optional

        Returns
        -------
        prob_hist_plot: ProbabilityHistogram
        """
        epoch_start -= 1
        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        self._prob_hist_data = ProbHistogramData(prob=self.predict_proba(epoch_start, epoch_end), target=self.targets)
        self._prob_hist_plot = ProbabilityHistogram(ax_negative, ax_positive).load_data(self._prob_hist_data)
        return self._prob_hist_plot

    def build_feature_space(self, ax, layer_name, grid_points= 1000, xlim=(-1, 1), ylim=(-1, 1), epoch_start=1,
                            epoch_end=-1):
        """

        Parameters
        ----------
        ax: AxesSubplot

        layer_name: String

        grid_points: int, optional

        xlim: tuple of ints, optional

        ylim: tuple of ints, optional

        epoch_start: int, optional

        epoch_end: int, optional

        Returns
        -------
        feature_space_plot: FeatureSpace
        """
        epoch_start -= 1
        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        X = self.inputs
        y = self.targets
        y_ind = y.squeeze().argsort()
        X = X.squeeze()[y_ind].reshape(X.shape)

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
        self._feature_space_data = FeatureSpaceData(line=line_data, bent_line=bent_line_data, prediction=bent_preds)
        self._feature_space_plot =  FeatureSpace(ax).load_data(self._feature_space_data)
        return self._feature_space_plot
