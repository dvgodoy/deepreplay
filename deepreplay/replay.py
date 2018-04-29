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
        # Set learning phase to TEST
        self.learning_phase = TEST_MODE

        # If not informed, defaults to '_model' suffix
        if model_filename == '':
            model_filename = '{}_model.h5'.format(group_name)

        # Loads Keras model
        self.model = load_model(model_filename)
        # Loads ReplayData file
        self.replay_data = h5py.File('{}'.format(replay_filename), 'r')
        self.group_name = group_name
        self.group = self.replay_data[self.group_name]

        # Retrieves some basic information from the replay data
        self.inputs = self.group['inputs'][:]
        self.targets = self.group['targets'][:]
        self.n_epochs = self.group.attrs['n_epochs']
        self.n_layers = self.group.attrs['n_layers']
        # Retrieves weights as a list, each element being one epoch
        self.weights = self._retrieve_weights()

        # Gets Tensors for the weights in the same order as the layers
        # Keras' model.weights returns the Tensors in a different order!
        self._model_weights = [w for layer in self.model.layers for w in layer.weights]

        ### Functions
        # Keras function to get the outputs, given inputs and weights
        self._get_output = K.function(inputs=[K.learning_phase()] + self.model.inputs + self._model_weights,
                                      outputs=[self.model.layers[-1].output])
        # Keras function to get the loss and metrics, given inputs, targets, weights and sample weights
        self._get_metrics = K.function(inputs=[K.learning_phase()] + self.model.inputs + self.model.targets +
                                              self._model_weights + self.model.sample_weights,
                                       outputs=[self.model.total_loss] + self.model.metrics_tensors)
        # Keras function to compute the binary cross entropy, given inputs, targets, weights and sample weights
        self._get_binary_crossentropy = K.function(inputs=[K.learning_phase()] + self.model.inputs +
                                                          self.model.targets + self._model_weights +
                                                          self.model.sample_weights,
                                                   outputs=[K.binary_crossentropy(self.model.targets[0],
                                                                                  self.model.outputs[0])])

        # Attributes for the visualizations - Data
        self._feature_space_data = None
        self._loss_hist_data = None
        self._loss_and_metric_data = None
        self._prob_hist_data = None
        # Attributes for the visualizations - Plot objects
        self._feature_space_plot = None
        self._loss_hist_plot = None
        self._loss_and_metric_plot = None
        self._prob_hist_plot = None

    def _retrieve_weights(self):
        # Generates ranges for the number of different weight arrays in each layer
        n_weights = [range(len(self.group['layer{}'.format(l)]))
                     for l in range(self.n_layers)]
        # Retrieves weights for each layer and sequence of weights
        weights = [np.array(self.group['layer{}'.format(l)]['weights{}'.format(w)])
                   for l, ws in enumerate(n_weights)
                   for w in ws]
        # Since initial weights are also saved, there are n_epochs + 1 elements in total
        return [[w[epoch] for w in weights] for epoch in range(self.n_epochs + 1)]

    def _make_function(self, inputs, layer):
        """Creates a Keras function to return the output of the
        `layer` argument, given inputs and weights
        """
        return K.function(inputs=[K.learning_phase()] + inputs + self._model_weights,
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
        """Returns corresponding metric as reported by Keras at the
        end of each epoch.

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

    def predict_proba(self, epoch_start=0, epoch_end=-1):
        """Generates class probability predictions for the inputs
        samples by epoch.

        Parameters
        ----------
        epoch_start: int, optional
            Initial epoch to return predicted probabilities for.
        epoch_end: int, optional
            Final epoch to return predicted probabilities for.

        Returns
        -------
        probas: ndarray
            An array of shape (n_epochs, n_samples, 2) of probabi-
            lity predictions.
        """
        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        probas = []
        # For each epoch, uses the corresponding weights
        for epoch in range(epoch_start, epoch_end + 1):
            weights = self.weights[epoch]
            probas.append(self._predict_proba(self.inputs, weights)[0])
        probas = np.array(probas)
        return probas

    def build_loss_histogram(self, ax, epoch_start=0, epoch_end=-1):
        """Builds a LossHistogram object to be used for plotting and
        animating.
        The underlying data, that is, the binary cross-entropy loss
        per epoch and sample, can be later accessed as the second
        element of the `loss_histogram` property.

        Only binary cross entropy loss is supported!

        Parameters
        ----------
        ax: AxesSubplot
            Subplot of a Matplotlib figure.
        epoch_start: int, optional
            First epoch to consider.
        epoch_end: int, optional
            Last epoch to consider.

        Returns
        -------
        loss_hist_plot: LossHistogram
            An instance of a LossHistogram object to make plots and
            animations.
        """
        if self.model.loss != 'binary_crossentropy':
            raise NotImplementedError("Only binary cross-entropy is supported!")

        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)
        binary_xentropy = []

        # For each epoch, uses the corresponding weights
        for epoch in range(epoch_start, epoch_end + 1):
            weights = self.weights[epoch]

            # Sample weights fixed to one!
            inputs = [self.learning_phase, self.inputs, self.targets] + weights + [np.ones(shape=self.inputs.shape[0])]
            binary_xentropy.append(self._get_binary_crossentropy(inputs=inputs)[0].squeeze())

        binary_xentropy = np.array(binary_xentropy)

        self._loss_hist_data = LossHistogramData(loss=binary_xentropy)
        self._loss_hist_plot = LossHistogram(ax).load_data(self._loss_hist_data)
        return self._loss_hist_plot

    def build_loss_and_metric(self, ax, metric_name, epoch_start=0, epoch_end=-1):
        """Builds a LossAndMetric object to be used for plotting and
        animating.
        The underlying data, that is, the loss and metric per epoch,
        can be later accessed as the second element of the
        `loss_and_metric` property.

        Parameters
        ----------
        ax: AxesSubplot
            Subplot of a Matplotlib figure.
        metric_name: String
            Metric to return values for.
        epoch_start: int, optional
            First epoch to consider.
        epoch_end: int, optional
            Last epoch to consider.

        Returns
        -------
        loss_and_metric_plot: LossAndMetric
            An instance of a LossAndMetric object to make plots and
            animations.
        """
        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        evaluations = []
        # For each epoch, uses the corresponding weights
        for epoch in range(epoch_start, epoch_end + 1):
            weights = self.weights[epoch]

            # Sample weights fixed to one!
            inputs = [self.learning_phase, self.inputs, self.targets] + weights + [np.ones(shape=self.inputs.shape[0])]
            evaluations.append(self._get_metrics(inputs=inputs))

        evaluations = np.array(evaluations)
        loss = evaluations[:, 0]
        try:
            metric = evaluations[:, self.model.metrics_names.index(metric_name)]
        except ValueError:
            metric = np.zeros(shape=(epoch_end - epoch_start + 1,))

        self._loss_and_metric_data = LossAndMetricData(loss=loss, metric=metric, metric_name=metric_name)
        self._loss_and_metric_plot = LossAndMetric(ax).load_data(self._loss_and_metric_data)
        return self._loss_and_metric_plot

    def build_probability_histogram(self, ax_negative, ax_positive, epoch_start=0, epoch_end=-1):
        """Builds a ProbabilityHistogram object to be used for plotting
        and animating.
        The underlying data, that is, the predicted probabilities
        and corresponding targets per epoch and sample, can be
        later accessed as the second element of the
        `probability_histogram` property.

        Only binary classification is supported!

        Parameters
        ----------
        ax_negative: AxesSubplot
            Subplot of a Matplotlib figure.
        ax_positive: AxesSubplot
            Subplot of a Matplotlib figure.
        epoch_start: int, optional
            First epoch to consider.
        epoch_end: int, optional
            Last epoch to consider.

        Returns
        -------
        prob_hist_plot: ProbabilityHistogram
            An instance of a ProbabilityHistogram object to make plots
            and animations.
        """
        if self.model.loss != 'binary_crossentropy':
            raise NotImplementedError("Only binary cross-entropy is supported!")

        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        self._prob_hist_data = ProbHistogramData(prob=self.predict_proba(epoch_start, epoch_end), target=self.targets)
        self._prob_hist_plot = ProbabilityHistogram(ax_negative, ax_positive).load_data(self._prob_hist_data)
        return self._prob_hist_plot

    def build_feature_space(self, ax, layer_name, contour_points=1000, xlim=(-1, 1), ylim=(-1, 1), scale_fixed=True,
                            display_grid=True, epoch_start=0, epoch_end=-1):
        """Builds a FeatureSpace object to be used for plotting and
        animating.
        The underlying data, that is, grid lines, inputs and contour
        lines, before and after the transformations, as well as the
        corresponding predictions for the contour lines, can be
        later accessed as the second element of the `feature_space`
        property.

        Only layers with 2 hidden units are supported!

        Parameters
        ----------
        ax: AxesSubplot
            Subplot of a Matplotlib figure.
        layer_name: String
            Layer to be used for building the space.
        contour_points: int, optional
            Number of points in each axis of the contour.
            Default is 1,000.
        xlim: tuple of ints, optional
            Boundaries for the X axis of the grid.
        ylim: tuple of ints, optional
            Boundaries for the Y axis of the grid.
        scaled_fixed: boolean, optional
            If True, axis scales are fixed to the maximum from beginning.
            Default is True.
        display_grid: boolean, optional
            If True, display grid lines (for 2-dimensional inputs).
            Default is True.
        epoch_start: int, optional
            First epoch to consider.
        epoch_end: int, optional
            Last epoch to consider.

        Returns
        -------
        feature_space_plot: FeatureSpace
            An instance of a FeatureSpace object to make plots and
            animations.
        """
        # Finds the layer by name,
        layer = self.model.get_layer(layer_name)
        assert layer.output_shape == (None, 2), 'Only layers with 2-dimensional outputs are supported!'

        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        X = self.inputs
        y = self.targets
        # Generates an indexing to fully separate negative and positive classes
        # It is not quite 'unshuffling', as it does not care about the order of the examples inside each class
        y_ind = y.squeeze().argsort()
        X = X.squeeze()[y_ind].reshape(X.shape)
        y = y.squeeze()[y_ind]

        input_dims = self.model.input_shape[-1]
        n_classes = len(np.unique(y))

        # Builds a 2D grid and the corresponding contour coordinates
        grid_lines = np.array([])
        contour_lines = np.array([])
        if input_dims == 2 and display_grid:
            grid_lines = build_2d_grid(xlim, ylim)
            contour_lines = build_2d_grid(xlim, ylim, contour_points, contour_points)

        # Creates Keras functions to get activations for the specified layer
        get_activations = self._make_function(self.model.inputs, layer)
        # Creates Keras function to get outputs of the last layer
        get_predictions = self._make_function(self.model.inputs, self.model.layers[-1])
        get_pred_from_act = self._make_function([layer.output], self.model.layers[-1])

        # Initializes "bent" variables, that is, the results of the transformations
        bent_lines = []
        bent_inputs = []
        bent_contour_lines = []
        bent_preds = []

        # For each epoch, uses the corresponding weights
        for epoch in range(epoch_start, epoch_end + 1):
            weights = self.weights[epoch]

            # Transforms the inputs
            inputs = [TEST_MODE, X] + weights
            bent_inputs.append(get_activations(inputs=inputs)[0])

            if input_dims == 2 and display_grid:
                # Transforms the grid lines
                inputs = [TEST_MODE, grid_lines.reshape(-1, 2)] + weights
                output_shape = (grid_lines.shape[:2]) + (-1,)
                bent_lines.append(get_activations(inputs=inputs)[0].reshape(output_shape))

                inputs = [TEST_MODE, contour_lines.reshape(-1, 2)] + weights
                output_shape = (contour_lines.shape[:2]) + (-1,)
                bent_contour_lines.append(get_activations(inputs=inputs)[0].reshape(output_shape))
                # Makes predictions for each point in the contour surface
                bent_preds.append((get_predictions(inputs=inputs)[0].reshape(output_shape) > .5).astype(np.int))

        bent_inputs = np.array(bent_inputs)

        if (input_dims > 2) or (not display_grid):
            xlim = (bent_inputs[:, :, 0].min(), bent_inputs[:, :, 0].max())
            ylim = (bent_inputs[:, :, 1].min(), bent_inputs[:, :, 1].max())
            grid_contour_lines = build_2d_grid(xlim, ylim, contour_points, contour_points)
            # For each epoch, uses the corresponding weights
            for epoch in range(epoch_start, epoch_end + 1):
                weights = self.weights[epoch]

                if not scale_fixed:
                    xlim = (bent_inputs[epoch, :, 0].min(), bent_inputs[epoch, :, 0].max())
                    ylim = (bent_inputs[epoch, :, 1].min(), bent_inputs[epoch, :, 1].max())
                    grid_contour_lines = build_2d_grid(xlim, ylim, contour_points, contour_points)
                # Transforms the contour lines
                bent_contour_lines.append(grid_contour_lines)

                inputs = [TEST_MODE, bent_contour_lines[-1].reshape(-1, 2)] + weights
                output_shape = (bent_contour_lines[-1].shape[:2]) + (-1,)
                bent_preds.append((get_pred_from_act(inputs=inputs)[0].reshape(output_shape) > .5).astype(np.int))

        # Makes lists into ndarrays and wrap them as namedtuples
        bent_lines = np.array(bent_lines)
        bent_contour_lines = np.array(bent_contour_lines)
        bent_preds = np.array(bent_preds)

        line_data = FeatureSpaceLines(grid=grid_lines, input=X, contour=contour_lines)
        bent_line_data = FeatureSpaceLines(grid=bent_lines, input=bent_inputs, contour=bent_contour_lines)
        self._feature_space_data = FeatureSpaceData(line=line_data, bent_line=bent_line_data,
                                                    prediction=bent_preds, target=y)

        # Creates a FeatureSpace plot object and load data into it
        self._feature_space_plot =  FeatureSpace(ax, scale_fixed).load_data(self._feature_space_data)
        return self._feature_space_plot
