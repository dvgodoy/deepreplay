from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from collections import namedtuple
from matplotlib import animation
matplotlib.rcParams['animation.writer'] = 'ffmpeg'
sns.set_style('white')

FeatureSpaceData = namedtuple('FeatureSpaceData', ['line', 'bent_line', 'prediction', 'target'])
FeatureSpaceLines = namedtuple('FeatureSpaceLines', ['grid', 'input', 'contour'])
LossAndMetricData = namedtuple('LossAndMetricData', ['loss', 'metric', 'metric_name'])
ProbHistogramData = namedtuple('ProbHistogramData', ['prob', 'target'])
LossHistogramData = namedtuple('LossHistogramData', ['loss'])

def build_2d_grid(xlim, ylim, n_lines=11, n_points=1000):
    """Returns a 2D grid of boundaries given by `xlim` and `ylim`,
     composed of `n_lines` evenly spaced lines of `n_points` each.

    Parameters
    ----------
    xlim : tuple of 2 ints
        Boundaries for the X axis of the grid.
    ylim : tuple of 2 ints
        Boundaries for the Y axis of the grid.
    n_lines : int, optional
        Number of grid lines. Default is 11.
        If n_lines equals n_points, the grid can be used as
        coordinates for the surface of a contourplot.
    n_points: int, optional
        Number of points in each grid line. Default is 1,000.

    Returns
    -------
    lines : ndarray
        For the cases where n_lines is less than n_points, it
        returns an array of shape (2 * n_lines, n_points, 2)
        containing both vertical and horizontal lines of the grid.
        If n_lines equals n_points, it returns an array of shape
        (n_points, n_points, 2), containing all evenly spaced
        points inside the grid boundaries.
    """
    xs = np.linspace(*xlim, num=n_lines)
    ys = np.linspace(*ylim, num=n_points)
    x0, y0 = np.meshgrid(xs, ys)
    lines_x0 = np.atleast_3d(x0.transpose())
    lines_y0 = np.atleast_3d(y0.transpose())

    xs = np.linspace(*xlim, num=n_points)
    ys = np.linspace(*ylim, num=n_lines)
    x1, y1 = np.meshgrid(xs, ys)
    lines_x1 = np.atleast_3d(x1)
    lines_y1 = np.atleast_3d(y1)

    vertical_lines = np.concatenate([lines_x0, lines_y0], axis=2)
    horizontal_lines = np.concatenate([lines_x1, lines_y1], axis=2)

    if n_lines != n_points:
        lines = np.concatenate([vertical_lines, horizontal_lines], axis=0)
    else:
        lines = vertical_lines

    return lines

def compose_animations(objects, epoch_start=0, epoch_end=-1, title=''):
    """Compose a single animation from several objects associated with
    subplots of a single figure.

    Parameters
    ----------
    objects: list of plot objects
        Plot objects returned using one of the 'build' methods of the
        Replay class. All the corresponding subplots associated with
        the objects must belong to the same figure.
    epoch_start: int, optional
        Epoch to start the animation from.
    epoch_end: int, optional
        Epoch to end the animation.
    title: String, optional
        Text to be used in the title, preceding the epoch information.

    Returns
    -------
    anim: FuncAnimation
        Composed animation function for all objects / subplots.
    """
    assert len(objects) > 1, 'Cannot compose using a single plot!'
    assert len(set([obj.fig for obj in objects])) == 1, 'All plots must belong to the same figure!'

    fig = objects[0].fig
    if epoch_end == -1:
        epoch_end = min([obj.n_epochs for obj in objects])

    if len(title):
        title += ' - '

    def update(i, objects, epoch_start=0):
        artists = []
        for obj in objects:
            artists += getattr(obj.__class__, '_update')(i, obj, epoch_start)
            for ax, ax_title in zip(obj.axes, obj.title):
                ax.set_title(ax_title)

        obj.fig.suptitle('{}Epoch {}'.format(title, i + epoch_start), fontsize=14)
        obj.fig.tight_layout()
        obj.fig.subplots_adjust(top=0.9)
        return artists

    anim = animation.FuncAnimation(fig, update,
                                   fargs=(objects, epoch_start),
                                   frames=(epoch_end - epoch_start),
                                   blit=True)
    return anim

def compose_plots(objects, epoch, title=''):
    """Compose a single plot from several objects associated with
    subplots of a single figure.

    Parameters
    ----------
    objects: list of plot objects
        Plot objects returned using one of the 'build' methods of the
        Replay class. All the corresponding subplots associated with
        the objects must belong to the same figure.
    epoch: int
        Epoch to use for the plotting.
    title: String, optional
        Text to be used in the title, preceding the epoch information.

    Returns
    -------
    fig: figure
        Figure which contains all subplots.
    """
    assert len(objects) > 1, 'Cannot compose using a single plot!'
    assert len(set([obj.fig for obj in objects])) == 1, 'All plots must belong to the same figure!'

    fig = objects[0].fig
    epoch_end = min([obj.n_epochs for obj in objects])
    epoch = min(epoch, epoch_end)

    for obj in objects:
        getattr(obj.__class__, '_update')(epoch, obj)
        for ax, ax_title in zip(obj.axes, obj.title):
            ax.set_title(ax_title)

    if len(title):
        title += ' - '
    fig.suptitle('{}Epoch {}'.format(title, epoch), fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    return fig

class Basic(object):
    """Basic plot class, NOT to be instantiated directly.
    """
    def __init__(self, ax):
        self._title = ''
        self.n_epochs = 0

        self.ax = ax
        self.ax.clear()
        self.fig = ax.get_figure()

    @property
    def title(self):
        return self._title if isinstance(self._title, tuple) else (self._title,)

    @property
    def axes(self):
        return (self.ax,)

    def load_data(self, **kwargs):
        self._prepare_plot()
        return self

    def _prepare_plot(self):
        pass

    @staticmethod
    def _update(i, object, epoch_start=0):
        pass

    def plot(self, epoch):
        """Plots data at a given epoch.

        Parameters
        ----------
        epoch: int
            Epoch to use for the plotting.

        Returns
        -------
        fig: figure
            Figure containing the plot.
        """
        self.__class__._update(epoch, self)
        self.fig.tight_layout()
        return self.fig

    def animate(self, epoch_start=0, epoch_end=-1):
        """Animates plotted data from `epoch_start` to `epoch_end`.

        Parameters
        ----------
        epoch_start: int, optional
            Epoch to start the animation from.
        epoch_end: int, optional
            Epoch to end the animation.

        Returns
        -------
        anim: FuncAnimation
            Animation function for the data.
        """
        if epoch_end == -1:
            epoch_end = self.n_epochs

        anim = animation.FuncAnimation(self.fig, self.__class__._update,
                                       fargs=(self, epoch_start),
                                       frames=(epoch_end - epoch_start),
                                       blit=True)
        return anim

class FeatureSpace(Basic):
    """Creates an instance of a FeatureSpace object to make plots
    and animations.

    Parameters
    ----------
    ax: AxesSubplot
        Subplot of a Matplotlib figure.
    scaled_fixed: boolean, optional
        If True, axis scales are fixed to the maximum from beginning.
        Default is True.
    """
    def __init__(self, ax, scale_fixed=True):
        super(FeatureSpace, self).__init__(ax)
        self.scale_fixed = scale_fixed
        self.contour = None
        self.bent_inputs = None
        self.bent_lines = None
        self.bent_contour_lines = None
        self.grid_lines = None
        self.contour_lines = None
        self.predictions = None
        self.targets = None

        self.n_inputs = 0

        self.lines = []
        self.points = []

    def load_data(self, feature_space_data):
        """ Loads feature space data as computed in Replay class.

        Parameters
        ----------
        feature_space_data: FeatureSpaceData
            Namedtuple containing information about original grid
            lines, data points and predictions.

        Returns
        -------
        self: FeatureSpace
            Returns the FeatureSpace instance itself.
        """
        self.predictions = feature_space_data.prediction
        self.targets = feature_space_data.target
        self.grid_lines, self.inputs, self.contour_lines = feature_space_data.line
        self.bent_lines, self.bent_inputs, self.bent_contour_lines = feature_space_data.bent_line

        self.n_epochs, _, self.n_inputs = self.bent_inputs.shape

        self.classes = np.unique(self.targets)
        self.bent_inputs = [self.bent_inputs[:, self.targets == target, :] for target in self.classes]

        self._prepare_plot()
        return self

    def _prepare_plot(self):
        if self.scale_fixed:
            xlim = [self.bent_contour_lines[:, :, :, 0].min(), self.bent_contour_lines[:, :, :, 0].max()]
            ylim = [self.bent_contour_lines[:, :, :, 1].min(), self.bent_contour_lines[:, :, :, 1].max()]
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        self.ax.set_xlabel(r"$x_1$", fontsize=14)
        self.ax.set_ylabel(r"$x_2$", fontsize=14, rotation=0)

        self.lines = []
        self.points = []
        for c in range(self.grid_lines.shape[0]):
            line, = self.ax.plot([], [], linewidth=0.5, color='k')
            self.lines.append(line)
        for c in range(len(self.classes)):
            point = self.ax.scatter([], [])
            self.points.append(point)

        contour_x = self.bent_contour_lines[0, :, :, 0]
        contour_y = self.bent_contour_lines[0, :, :, 1]
        self.contour = self.ax.contourf(contour_x, contour_y, np.zeros(shape=(len(contour_x), len(contour_y))),
                              cmap=plt.cm.brg, alpha=0.3, levels=np.linspace(0, 1, 8))

    @staticmethod
    def _update(i, fs, epoch_start=0):
        epoch = i + epoch_start
        fs.ax.set_title('Epoch: {}'.format(epoch))
        if not fs.scale_fixed:
            xlim = [fs.bent_contour_lines[epoch, :, :, 0].min(), fs.bent_contour_lines[epoch, :, :, 0].max()]
            ylim = [fs.bent_contour_lines[epoch, :, :, 1].min(), fs.bent_contour_lines[epoch, :, :, 1].max()]
            fs.ax.set_xlim(xlim)
            fs.ax.set_ylim(ylim)

        if len(fs.lines):
            line_coords = fs.bent_lines[epoch].transpose()

        for c, line in enumerate(fs.lines):
            line.set_data(*line_coords[:, :, c])

        colors = ['b', 'g']
        input_coords = [coord[epoch].transpose() for coord in fs.bent_inputs]
        for c in range(len(fs.points)):
            fs.points[c].remove()
            fs.points[c] = fs.ax.scatter(*input_coords[c], marker='o', color=colors[c], s=10)

        for c in fs.contour.collections:
            c.remove()  # removes only the contours, leaves the rest intact

        fs.contour = fs.ax.contourf(fs.bent_contour_lines[epoch, :, :, 0],
                                    fs.bent_contour_lines[epoch, :, :, 1],
                                    fs.predictions[epoch].squeeze(),
                                    cmap=plt.cm.brg, alpha=0.3, levels=np.linspace(0, 1, 8))

        fs.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        fs.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        fs.ax.locator_params(tight=True, nbins=7)
        return fs.lines


class ProbabilityHistogram(Basic):
    """Creates an instance of a ProbabilityHistogram object to make
    plots and animations.

    Parameters
    ----------
    ax1: AxesSubplot
        Subplot of a Matplotlib figure, for the negative cases.
    ax2: AxesSubplot
        Subplot of a Matplotlib figure, for the positive cases.
    """
    def __init__(self, ax1, ax2):
        self._title = ('Negative Cases', 'Positive Cases')
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax1.clear()
        self.ax2.clear()
        self.fig = ax1.get_figure()
        self.line = ax1.plot([], [])

        self.proba = None
        self.targets = None
        self.bins = np.linspace(0, 1, 11)

    @property
    def axes(self):
        return (self.ax1, self.ax2)

    def load_data(self, prob_histogram_data):
        """ Loads probability histogram data as computed in Replay
        class.

        Parameters
        ----------
        prob_histogram_data: ProbHistogramData
            Namedtuple containing information about classification
            probabilities and targets.

        Returns
        -------
        self: ProbabilityHistogram
            Returns the ProbabilityHistogram instance itself.
        """
        self.proba, self.targets = prob_histogram_data

        self.n_epochs = self.proba.shape[0]
        self._prepare_plot()
        return self

    def _prepare_plot(self):
        pass

    @staticmethod
    def _update(i, ph, epoch_start=0):
        epoch = i + epoch_start

        correct = ((ph.proba[epoch] > .5) == ph.targets)
        tn = ph.proba[epoch, (ph.targets == 0) & correct]
        fn = ph.proba[epoch, (ph.targets == 0) & ~correct]
        tp = ph.proba[epoch, (ph.targets == 1) & correct]
        fp = ph.proba[epoch, (ph.targets == 1) & ~correct]

        for ax in (ph.ax1, ph.ax2):
            ax.clear()

        ph.ax1.set_title('{} - Epoch: {}'.format(ph.title[0], epoch))
        ph.ax1.set_ylim([0, (ph.targets == 0).sum()])
        ph.ax1.set_xlabel('Probability')
        ph.ax1.set_ylabel('# of Cases')
        ph.ax1.hist(tn, bins=ph.bins, color='k', alpha=.4)
        ph.ax1.hist(fn, bins=ph.bins, color='r', alpha=.5)

        ph.ax2.set_title('{} - Epoch: {}'.format(ph.title[1], epoch))
        ph.ax2.set_ylim([0, (ph.targets == 1).sum()])
        ph.ax2.set_xlabel('Probability')
        ph.ax2.set_ylabel('# of Cases')
        ph.ax2.hist(tp, bins=ph.bins, color='k', alpha=.4)
        ph.ax2.hist(fp, bins=ph.bins, color='r', alpha=.5)

        ph.ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        ph.ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        ph.ax1.locator_params(tight=True, nbins=3)
        ph.ax2.locator_params(tight=True, nbins=3)

        return ph.line

class LossAndMetric(Basic):
    """Creates an instance of a LossAndMetric object to make plots
    and animations.

    Parameters
    ----------
    ax: AxesSubplot
        Subplot of a Matplotlib figure.
    """
    def __init__(self, ax):
        super(LossAndMetric, self).__init__(ax)
        self.ax2 = self.ax.twinx()
        self.line1 = None
        self.line2 = None
        self.point1 = None
        self.point2 = None
        self.metric = None
        self.metric_name = ''

    def load_data(self, loss_and_metric_data):
        """ Loads loss and metric data as computed in Replay class.

        Parameters
        ----------
        loss_and_metric_data: LossAndMetricData
            Namedtuple containing information about loss and a
            given metric.

        Returns
        -------
        self: LossAndMetric
            Returns the LossAndMetric instance itself.
        """
        self.loss, self.metric, self.metric_name = loss_and_metric_data
        self._title = '{} / Loss'.format(self.metric_name)

        self.n_epochs = self.loss.shape[0]
        self._prepare_plot()
        return self

    def _prepare_plot(self):
        self.ax.set_xlim([0, self.n_epochs])
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylim([0, 1.01 * self.metric.max()])
        if self.metric_name == 'acc':
            self.ax.set_ylim([0, 1.01])
        self.ax.set_ylabel(self.metric_name)

        self.ax2.set_xlim([0, self.n_epochs])
        self.ax2.set_ylim([0, 1.01 * self.loss.max()])
        self.ax2.set_ylabel('Loss')

        self.line1, = self.ax.plot([], [], color='k')
        self.line2, = self.ax2.plot([], [], color='r')
        self.point1 = self.ax.scatter([], [], marker='o')
        self.point2 = self.ax2.scatter([], [], marker='o')

        self.ax.legend((self.line1, self.line2), (self.metric_name, 'Loss'), loc=3)

    @staticmethod
    def _update(i, lm, epoch_start=0):
        epoch = i + epoch_start
        lm.ax.set_title('{} - Epoch: {}'.format(lm.title[0], epoch))

        lm.line1.set_data(np.arange(0, epoch + 1), lm.metric[:epoch + 1])
        lm.line2.set_data(np.arange(0, epoch + 1), lm.loss[:epoch + 1])
        lm.point1.remove()
        lm.point1 = lm.ax.scatter(epoch, lm.metric[epoch], marker='o', color='k')
        lm.point2.remove()
        lm.point2 = lm.ax2.scatter(epoch, lm.loss[epoch], marker='o', color='r')

        lm.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        lm.ax.locator_params(tight=True, nbins=3)

        return lm.line1, lm.line2

class LossHistogram(Basic):
    """Creates an instance of a LossHistogram object to make plots
    and animations.

    Parameters
    ----------
    ax: AxesSubplot
        Subplot of a Matplotlib figure.
    """
    def __init__(self, ax):
        super(LossHistogram, self).__init__(ax)
        self.losses = None
        self._title = 'Losses'

    def __calc_scale(self, margin):
        """ Computes the bins partition for the histogram plot based on the loss range.
        """
        loss_limits = np.array([self.losses.squeeze().min(), self.losses.squeeze().max()])
        loss_range = np.diff(loss_limits)[0]
        exponent = np.floor(np.log10(loss_range))
        magnitude = np.power(10, exponent)
        loss_limits = np.round(loss_limits + np.array([-margin, margin]) * magnitude, exponent.astype(np.int) + 1)
        intervals = (np.diff(loss_limits)[0] / magnitude + 1).astype(np.int)
        while 10 > intervals > 1:
            intervals = (intervals - 1) * 2 + 1
        loss_scale = np.linspace(max(0.0, loss_limits[0]), loss_limits[1], intervals)
        return loss_scale

    def load_data(self, loss_hist_data):
        """ Loads loss histogram data as computed in Replay class.

        Parameters
        ----------
        loss_hist_data: LossHistogramData
            Namedtuple containing information about example's losses.

        Returns
        -------
        self: LossHistogram
            Returns the LossHistogram instance itself.
        """
        self.losses, = loss_hist_data
        self.bins = self.__calc_scale(margin=0)

        self.n_epochs = self.losses.shape[0]
        self._prepare_plot()
        return self

    def _prepare_plot(self):
        self.line = self.ax.plot([], [])

    @staticmethod
    def _update(i, lh, epoch_start=0):
        epoch = i + epoch_start

        lh.ax.clear()

        lh.ax.set_title('{} - Epoch: {}'.format(lh.title[0], epoch))
        lh.ax.set_ylim([0, lh.losses.shape[1]])
        lh.ax.set_xlabel('Loss')
        lh.ax.set_ylabel('# of Cases')
        lh.ax.hist(lh.losses.squeeze()[i], bins=lh.bins, color='k', alpha=.4)

        lh.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        lh.ax.locator_params(tight=True, nbins=4)

        return lh.line