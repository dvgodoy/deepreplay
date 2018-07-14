from __future__ import division
import numpy as np
import h5py
import keras.backend as K
from keras.models import load_model
from .plot import (
    build_2d_grid, FeatureSpace, ProbabilityHistogram, LossHistogram, LossAndMetric, LayerViolins
)
from .plot import (
    FeatureSpaceData, FeatureSpaceLines, ProbHistogramData, LossHistogramData, LossAndMetricData, LayerViolinsData
)
from .utils import make_batches, slice_arrays
from itertools import groupby
from operator import itemgetter

TRAINING_MODE = 1
TEST_MODE = 0
ACTIVATIONS = ['softmax', 'relu', 'elu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'softplus', 'softsign', 'selu']
Z_OPS = ['BiasAdd', 'MatMul', 'Add', 'Sub', 'Mul', 'Maximum', 'Minimum', 'RealDiv', 'ExpandDims']


class Replay(object):
    """Creates an instance of Replay, to process information collected
    by the callback and generate data to feed the supported visualizations.

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

    weights_violins: (LayerViolins, LayerViolinsData)
        LayerViolins object to be used for plotting and animating;
        namedtuple containing information about weights values
        per layer.

    activations_violins: (LayerViolins, LayerViolinsData)
        LayerViolins object to be used for plotting and animating;
        namedtuple containing information about activation values
        per layer.

    zvalues_violins: (LayerViolins, LayerViolinsData)
        LayerViolins object to be used for plotting and animating;
        namedtuple containing information about Z-values per layer.

    gradients_violins: (LayerViolins, LayerViolinsData)
        LayerViolins object to be used for plotting and animating;
        namedtuple containing information about gradient values
        per layer.

    weights_std: ndarray
        Standard deviation of the weights per layer.

    gradients_std: ndarray
        Standard deivation of the gradients per layer.

    training_loss: ndarray
        An array of shape (n_epochs, ) with training loss as reported
        by Keras at the end of each epoch.

    learning_rate: ndarray
        An array of shape (n_epochs, ) with learning rate as reported
        by Keras at the beginning of each epoch.
    """
    def __init__(self, replay_filename, group_name, model_filename=''):
        # Set learning phase to TEST
        self.learning_phase = TEST_MODE

        # Loads ReplayData file
        self.replay_data = h5py.File('{}'.format(replay_filename), 'r')
        try:
            self.group = self.replay_data[group_name]
        except KeyError:
            self.group = self.replay_data[group_name + '_init']
            group_name += '_init'

        self.group_name = group_name

        # If not informed, defaults to '_model' suffix
        if model_filename == '':
            model_filename = '{}_model.h5'.format(group_name)
        # Loads Keras model
        self.model = load_model(model_filename)

        # Retrieves some basic information from the replay data
        self.inputs = self.group['inputs'][:]
        self.targets = self.group['targets'][:]
        self.n_epochs = self.group.attrs['n_epochs']
        self.n_layers = self.group.attrs['n_layers']

        # Generates ranges for the number of different weight arrays in each layer
        self.n_weights = [range(len(self.group['layer{}'.format(l)])) for l in range(self.n_layers)]

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

        # Keras function to compute the gradients for trainable weights, given inputs, targets, weights and
        # sample weights
        self.__trainable_weights = [w for layer in self.model.layers
                                    for w in layer.trainable_weights
                                    if layer.trainable and ('bias' not in w.op.name)]
        self.__trainable_gradients = self.model.optimizer.get_gradients(self.model.total_loss, self.__trainable_weights)
        self._get_gradients = K.function(inputs=[K.learning_phase()] + self.model.inputs + self.model.targets +
                                                self._model_weights + self.model.sample_weights,
                                         outputs=self.__trainable_gradients)

        def get_z_op(layer):
            op = layer.output.op
            if op.type in Z_OPS:
                return layer.output
            else:
                op_layer_name = op.name.split('/')[0]
                for input in op.inputs:
                    input_layer_name = input.name.split('/')[0]
                    if (input.op.type in Z_OPS) and (op_layer_name == input_layer_name):
                        return input
                return None

        __z_layers = np.array([i for i, layer in enumerate(self.model.layers) if get_z_op(layer) is not None])
        __act_layers = np.array([i for i, layer in enumerate(self.model.layers)
                               if layer.output.op.type.lower() in ACTIVATIONS])
        __z_layers = np.array([__z_layers[np.argmax(layer < __z_layers) - 1] for layer in __act_layers])
        self.z_act_layers = [self.model.layers[i].name for i in __z_layers]

        self._z_layers = ['inputs'] + [self.model.layers[i].name for i in __z_layers]
        self._z_tensors = [K.identity(self.model.inputs)] + list(filter(lambda t: t is not None,
                                                          [get_z_op(self.model.layers[i]) for i in __z_layers]))

        self._activation_layers = ['inputs'] + [self.model.layers[i].name for i in __act_layers]
        self._activation_tensors = [K.identity(self.model.inputs)] + [self.model.layers[i].output for i in __act_layers]

        # Keras function to compute the Z values given inputs and weights
        self._get_zvalues = K.function(inputs=[K.learning_phase()] + self.model.inputs + self._model_weights,
                                       outputs=self._z_tensors)
        # Keras function to compute the activation values given inputs and weights
        self._get_activations = K.function(inputs=[K.learning_phase()] + self.model.inputs + self._model_weights,
                                           outputs=self._activation_tensors)

        # Gets names of all layers with arrays of weights of lengths 1 (no biases) or 2 (with biases)
        # Layers without weights (e.g. Activation, BatchNorm) are not included
        self.weights_layers = [layer.name for layer, weights in zip(self.model.layers, self.n_weights)
                               if len(weights) in (1, 2)]

        # Attributes for the visualizations - Data
        self._feature_space_data = None
        self._loss_hist_data = None
        self._loss_and_metric_data = None
        self._prob_hist_data = None
        self._decision_boundary_data = None
        self._weights_violins_data = None
        self._activations_violins_data = None
        self._zvalues_violins_data = None
        self._gradients_data = None
        # Attributes for the visualizations - Plot objects
        self._feature_space_plot = None
        self._loss_hist_plot = None
        self._loss_and_metric_plot = None
        self._prob_hist_plot = None
        self._decision_boundary_plot = None
        self._weights_violins_plot = None
        self._activations_violins_plot = None
        self._zvalues_violins_plot = None
        self._gradients_plot = None

    def _make_batches(self, seed):
        inputs = self.inputs[:]
        targets = self.targets[:]

        np.random.seed(seed)
        np.random.shuffle(inputs)
        np.random.shuffle(targets)
        num_training_samples = inputs.shape[0]

        batches = make_batches(num_training_samples, self.params['batch_size'])
        index_array = np.arange(num_training_samples)

        inputs_batches = []
        targets_batches = []
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            inputs_batch, targets_batch = slice_arrays([inputs, targets], batch_ids)
            inputs_batches.append(inputs_batch)
            targets_batches.append(targets_batch)

        return inputs_batches, targets_batches

    @staticmethod
    def __assign_gradients_to_layers(layers, gradients):
        return [list(list(zip(*g))[1]) for k, g in groupby(zip(layers, gradients), itemgetter(0))]

    def _retrieve_weights(self):
        # Retrieves weights for each layer and sequence of weights
        weights = [np.array(self.group['layer{}'.format(l)]['weights{}'.format(w)])
                   for l, ws in enumerate(self.n_weights)
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
    def decision_boundary(self):
        return self._decision_boundary_plot, self._decision_boundary_data

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
    def weights_violins(self):
        return self._weights_violins_plot, self._weights_violins_data

    @property
    def activations_violins(self):
        return self._activations_violins_plot, self._activations_violins_data

    @property
    def zvalues_violins(self):
        return self._zvalues_violins_plot, self._zvalues_violins_data

    @property
    def gradients_violins(self):
        return self._gradients_plot, self._gradients_data

    @staticmethod
    def __calc_std(values):
        return np.array([[layer.std() for layer in epoch] for epoch in values])

    @property
    def weights_std(self):
        std = None
        if self._weights_violins_data is not None:
            weights = self._weights_violins_data.values
            std = Replay.__calc_std(weights)
        return std

    @property
    def gradients_std(self):
        std = None
        if self._gradients_data is not None:
            gradients = self._gradients_data.values
            std = Replay.__calc_std(gradients)
        return std

    @property
    def training_loss(self):
        return self.group['loss'][:]

    @property
    def learning_rate(self):
        return self.group['lr'][:]

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

    def build_gradients(self, ax, layer_names=None, exclude_outputs=True, epoch_start=0, epoch_end=-1):
        """Builds a LayerViolins object to be used for plotting and
        animating.

        Parameters
        ----------
        ax: AxesSubplot
            Subplot of a Matplotlib figure.
        layer_names: list of Strings, optional
            If informed, plots only the listed layers.
        exclude_outputs: boolean, optional
            If True, excludes distribution of output layer. Default is True.
            If `layer_names` is informed, `exclude_outputs` is ignored.
        epoch_start: int, optional
            First epoch to consider.
        epoch_end: int, optional
            Last epoch to consider.

        Returns
        -------
        gradients_plot: LayerViolins
            An instance of a LayerViolins object to make plots and
            animations.
        """
        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        gradient_names = [layer.name for layer in self.model.layers for w in layer.trainable_weights
                          if layer.trainable and ('bias' not in w.op.name)]
        gradients = []
        # For each epoch, uses the corresponding weights
        for epoch in range(epoch_start, epoch_end + 1):
            weights = self.weights[epoch]

            # Sample weights fixed to one!
            inputs = [self.learning_phase, self.inputs, self.targets] + weights + [np.ones(shape=self.inputs.shape[0])]
            grad = [w for v in Replay.__assign_gradients_to_layers(gradient_names, self._get_gradients(inputs=inputs))
                    for w in v]
            gradients.append(grad)

        if layer_names is None:
            layer_names = self.weights_layers
            if exclude_outputs:
                layer_names = layer_names[:-1]

        self._gradients_data = LayerViolinsData(names=gradient_names, values=gradients, layers=self.weights_layers,
                                                selected_layers=layer_names)
        if ax is None:
            self._gradients_plot = None
        else:
            self._gradients_plot = LayerViolins(ax, 'Gradients').load_data(self._gradients_data)
        return self._gradients_plot

    def build_outputs(self, ax, before_activation=False, layer_names=None, include_inputs=True,
                      exclude_outputs=True, epoch_start=0, epoch_end=-1):
        """Builds a LayerViolins object to be used for plotting and
        animating.

        Parameters
        ----------
        ax: AxesSubplot
            Subplot of a Matplotlib figure.
        before_activation: Boolean, optional
            If True, returns Z-values, that is, before applying
            the activation function.
        layer_names: list of Strings, optional
            If informed, plots only the listed layers.
        include_inputs: boolean, optional
            If True, includes distribution of inputs. Default is True.
        exclude_outputs: boolean, optional
            If True, excludes distribution of output layer. Default is True.
            If `layer_names` is informed, `exclude_outputs` is ignored.
        epoch_start: int, optional
            First epoch to consider.
        epoch_end: int, optional
            Last epoch to consider.

        Returns
        -------
        activations_violins_plot/zvalues_violins_plot: LayerViolins
            An instance of a LayerViolins object to make plots and
            animations.
        """
        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        if before_activation:
            title = 'Z-values'
            names = self._z_layers
        else:
            title = 'Activations'
            names = self._activation_layers
        outputs = []
        # For each epoch, uses the corresponding weights
        for epoch in range(epoch_start, epoch_end + 1):
            weights = self.weights[epoch]
            inputs = [self.learning_phase, self.inputs] + weights
            if before_activation:
                outputs.append(self._get_zvalues(inputs=inputs))
            else:
                outputs.append(self._get_activations(inputs=inputs))

        if layer_names is None:
            layer_names = self.z_act_layers
            if exclude_outputs:
                layer_names = layer_names[:-1]
        if include_inputs:
            layer_names = ['inputs'] + layer_names

        data = LayerViolinsData(names=names, values=outputs, layers=self.z_act_layers, selected_layers=layer_names)
        if ax is None:
            plot = None
        else:
            plot = LayerViolins(ax, title).load_data(data)
        if before_activation:
            self._zvalues_violins_data = data
            self._zvalues_violins_plot = plot
        else:
            self._activations_violins_data = data
            self._activations_violins_plot = plot
        return plot

    def build_weights(self, ax, layer_names=None, exclude_outputs=True, epoch_start=0, epoch_end=-1):
        """Builds a LayerViolins object to be used for plotting and
        animating.

        Parameters
        ----------
        ax: AxesSubplot
            Subplot of a Matplotlib figure.
        layer_names: list of Strings, optional
            If informed, plots only the listed layers.
        exclude_outputs: boolean, optional
            If True, excludes distribution of output layer. Default is True.
            If `layer_names` is informed, `exclude_outputs` is ignored.
        epoch_start: int, optional
            First epoch to consider.
        epoch_end: int, optional
            Last epoch to consider.

        Returns
        -------
        weights_violins_plot: LayerViolins
            An instance of a LayerViolins object to make plots and
            animations.
        """
        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        names = [layer.name for layer, weights in zip(self.model.layers, self.n_weights) if len(weights) in (1, 2)]
        n_weights = [(i, len(weights)) for layer, weights in zip(self.model.layers, self.n_weights) for i in weights]

        weights = []
        # For each epoch, uses the corresponding weights
        for epoch in range(epoch_start, epoch_end + 1):
            # takes only the weights (i == 0), not the biases (i == 1)
            weights.append([w for w, (i, n) in zip(self.weights[epoch], n_weights) if (n in (1, 2)) and (i == 0)])

        if layer_names is None:
            layer_names = self.weights_layers
            if exclude_outputs:
                layer_names = layer_names[:-1]

        self._weights_violins_data = LayerViolinsData(names=names,
                                                      values=weights,
                                                      layers=self.weights_layers,
                                                      selected_layers=layer_names)
        if ax is None:
            self._weights_violins_plot = None
        else:
            self._weights_violins_plot = LayerViolins(ax, 'Weights').load_data(self._weights_violins_data)
        return self._weights_violins_plot

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

    def build_decision_boundary(self, ax, contour_points=1000, xlim=(-1, 1), ylim=(-1, 1), display_grid=True,
                                epoch_start=0, epoch_end=-1):
        """Builds a FeatureSpace object to be used for plotting and
        animating the raw inputs and the decision boundary.
        The underlying data, that is, grid lines, inputs and contour
        lines, as well as the corresponding predictions for the
        contour lines, can be later accessed as the second element of
        the  `decision_boundary` property.

        Only inputs with 2 dimensions are supported!

        Parameters
        ----------
        ax: AxesSubplot
            Subplot of a Matplotlib figure.
        contour_points: int, optional
            Number of points in each axis of the contour.
            Default is 1,000.
        xlim: tuple of ints, optional
            Boundaries for the X axis of the grid.
        ylim: tuple of ints, optional
            Boundaries for the Y axis of the grid.
        display_grid: boolean, optional
            If True, display grid lines (for 2-dimensional inputs).
            Default is True.
        epoch_start: int, optional
            First epoch to consider.
        epoch_end: int, optional
            Last epoch to consider.

        Returns
        -------
        decision_boundary_plot: FeatureSpace
            An instance of a FeatureSpace object to make plots and
            animations.
        """
        input_dims = self.model.input_shape[-1]
        assert input_dims == 2, 'Only layers with 2-dimensional inputs are supported!'

        if epoch_end == -1:
            epoch_end = self.n_epochs
        epoch_end = min(epoch_end, self.n_epochs)

        X = self.inputs
        y = self.targets

        y_ind = y.squeeze().argsort()
        X = X.squeeze()[y_ind].reshape(X.shape)
        y = y.squeeze()[y_ind]

        n_classes = len(np.unique(y))

        # Builds a 2D grid and the corresponding contour coordinates
        grid_lines = np.array([])
        if display_grid:
            grid_lines = build_2d_grid(xlim, ylim)

        contour_lines = build_2d_grid(xlim, ylim, contour_points, contour_points)
        get_predictions = self._make_function(self.model.inputs, self.model.layers[-1])

        bent_lines = []
        bent_inputs = []
        bent_contour_lines = []
        bent_preds = []
        # For each epoch, uses the corresponding weights
        for epoch in range(epoch_start, epoch_end + 1):
            weights = self.weights[epoch]

            bent_lines.append(grid_lines)
            bent_inputs.append(X)
            bent_contour_lines.append(contour_lines)

            inputs = [TEST_MODE, contour_lines.reshape(-1, 2)] + weights
            output_shape = (contour_lines.shape[:2]) + (-1,)
            # Makes predictions for each point in the contour surface
            bent_preds.append((get_predictions(inputs=inputs)[0].reshape(output_shape) > .5).astype(np.int))

        # Makes lists into ndarrays and wrap them as namedtuples
        bent_inputs = np.array(bent_inputs)
        bent_lines = np.array(bent_lines)
        bent_contour_lines = np.array(bent_contour_lines)
        bent_preds = np.array(bent_preds)

        line_data = FeatureSpaceLines(grid=grid_lines, input=X, contour=contour_lines)
        bent_line_data = FeatureSpaceLines(grid=bent_lines, input=bent_inputs, contour=bent_contour_lines)
        self._decision_boundary_data = FeatureSpaceData(line=line_data, bent_line=bent_line_data,
                                                        prediction=bent_preds, target=y)

        # Creates a FeatureSpace plot object and load data into it
        self._decision_boundary_plot =  FeatureSpace(ax, True).load_data(self._decision_boundary_data)
        return self._decision_boundary_plot

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
