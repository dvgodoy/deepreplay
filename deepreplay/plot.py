from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from collections import namedtuple
from matplotlib import animation
matplotlib.rcParams['animation.writer'] = 'avconv'
sns.set_style('white')

FeatureSpaceData = namedtuple('FeatureSpaceData', ['line', 'bent_line', 'prediction'])
FeatureSpaceLines = namedtuple('FeatureSpaceLines', ['grid', 'input', 'contour'])
LossAndMetricData = namedtuple('LossAndMetricData', ['loss', 'metric', 'metric_name'])
ProbHistogramData = namedtuple('ProbHistogramData', ['prob', 'target'])
LossHistogramData = namedtuple('LossHistogramData', ['loss'])

def build_2d_grid(xlim, ylim, xn=11, yn=1000):
    x0s = np.linspace(*xlim, num=xn)
    x1s = np.linspace(*ylim, num=yn)
    x0, x1 = np.meshgrid(x0s, x1s)

    lines_x0 = np.atleast_3d(x0.transpose())
    lines_x1 = np.atleast_3d(x1.transpose())

    if xn != yn:
        lines_x0, lines_x1 = np.concatenate([lines_x0, lines_x1]), np.concatenate([lines_x1, lines_x0])

    lines = np.concatenate([lines_x0, lines_x1], axis=2)
    return lines

def compose_animations(objects, epoch_start=1, epoch_end=-1, title=''):
    assert len(objects) > 1
    assert len(set([obj.fig for obj in objects])) == 1

    fig = objects[0].fig
    epoch_start -= 1
    if epoch_end == -1:
        epoch_end = min([obj.n_epochs for obj in objects])

    if len(title):
        title += ' - '

    def update(i, objects, epoch_start=0):
        artists = []
        for obj in objects:
            artists += getattr(obj.__class__, 'update')(i, obj, epoch_start)
            for ax, ax_title in zip(obj.axes, obj.title):
                ax.set_title(ax_title)

        obj.fig.suptitle('{}Epoch {}'.format(title, i + epoch_start + 1), fontsize=14)
        obj.fig.tight_layout()
        obj.fig.subplots_adjust(top=0.9)
        return artists

    anim = animation.FuncAnimation(fig, update,
                                   fargs=(objects, epoch_start),
                                   frames=(epoch_end - epoch_start),
                                   blit=True)
    return anim

def compose_plots(objects, epoch, title=''):
    assert len(objects) > 1
    assert len(set([obj.fig for obj in objects])) == 1

    fig = objects[0].fig
    epoch_end = min([obj.n_epochs for obj in objects])
    epoch = min(epoch, epoch_end)

    for obj in objects:
        getattr(obj.__class__, 'update')(epoch - 1, obj)
        for ax, ax_title in zip(obj.axes, obj.title):
            ax.set_title(ax_title)

    if len(title):
        title += ' - '
    fig.suptitle('{}Epoch {}'.format(title, epoch), fontsize=14)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    return fig

class Basic(object):
    def __init__(self, ax):
        self._title = ''
        self.n_epochs = 0

        self.ax = ax
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
    def update(i, object, epoch_start=0):
        pass

    def plot(self, epoch):
        self.__class__.update(epoch - 1, self)
        self.fig.tight_layout()
        return self.fig

    def animate(self, epoch_start=1, epoch_end=-1):
        epoch_start -= 1
        if epoch_end == -1:
            epoch_end = self.n_epochs

        anim = animation.FuncAnimation(self.fig, self.__class__.update,
                                       fargs=(self, epoch_start),
                                       frames=(epoch_end - epoch_start),
                                       blit=True)
        return anim

class FeatureSpace(Basic):
    def __init__(self, ax):
        super(FeatureSpace, self).__init__(ax)
        self.contour = None
        self.bent_inputs = None
        self.bent_lines = None
        self.bent_contour_lines = None
        self.inputs = None
        self.grid_lines = None
        self.contour_lines = None
        self.predictions = None

        self.n_inputs = 0

        self.lines = []
        self.points = []

    def load_data(self, feature_space_data):
        self.predictions = feature_space_data.prediction
        self.grid_lines, self.inputs, self.contour_lines = feature_space_data.line
        self.bent_lines, self.bent_inputs, self.bent_contour_lines = feature_space_data.bent_line

        self.n_epochs, _, self.n_inputs = self.bent_inputs.shape

        self.inputs = self.inputs.reshape(2, -1, self.n_inputs)
        self.bent_inputs = self.bent_inputs.reshape(self.n_epochs, 2, -1, self.n_inputs)

        self._prepare_plot()
        return self

    def _prepare_plot(self):
        xlim = [self.bent_lines[:, :, :, 0].min(), self.bent_lines[:, :, :, 0].max()]
        ylim = [self.bent_lines[:, :, :, 1].min(), self.bent_lines[:, :, :, 1].max()]
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        self.ax.set_xlabel(r"$x_1$", fontsize=14)
        self.ax.set_ylabel(r"$x_2$", fontsize=14, rotation=0)

        self.lines = []
        self.points = []
        for c in range(self.grid_lines.shape[0]):
            line, = self.ax.plot([], [], linewidth=0.5, color='k')
            self.lines.append(line)
        for c in range(self.inputs.shape[0]):
            point = self.ax.scatter([], [])
            self.points.append(point)

        contour_x = self.contour_lines[:, :, 0]
        contour_y = self.contour_lines[:, :, 1]
        self.contour = self.ax.contourf(contour_x, contour_y, np.zeros(shape=(len(contour_x), len(contour_y))),
                              cmap=plt.cm.brg, alpha=0.3, levels=np.linspace(0, 1, 8))

    @staticmethod
    def update(i, fs, epoch_start=0):
        epoch = i + epoch_start
        fs.ax.set_title('Epoch: {}'.format(epoch + 1))

        line_coords = fs.bent_lines[epoch].transpose()
        input_coords = fs.bent_inputs[epoch].transpose()

        for c, line in enumerate(fs.lines):
            line.set_data(*line_coords[:, :, c])

        colors = ['b', 'g']
        for c in range(len(fs.points)):
            fs.points[c].remove()
            fs.points[c] = fs.ax.scatter(*input_coords[:, :, c], marker='o', color=colors[c], s=10)

        for c in fs.contour.collections:
            c.remove()  # removes only the contours, leaves the rest intact

        fs.contour = fs.ax.contourf(fs.bent_contour_lines[epoch, :, :, 0],
                              fs.bent_contour_lines[epoch, :, :, 1],
                              fs.predictions[epoch].squeeze(),
                              cmap=plt.cm.brg, alpha=0.3, levels=np.linspace(0, 1, 8))

        return fs.lines


class ProbabilityHistogram(Basic):
    def __init__(self, ax1, ax2):
        self._title = ('Negative Cases', 'Positive Cases')
        self.ax1 = ax1
        self.ax2 = ax2
        self.fig = ax1.get_figure()
        self.line = ax1.plot([], [])

        self.proba = None
        self.targets = None
        self.bins = np.linspace(0, 1, 11)

    @property
    def axes(self):
        return (self.ax1, self.ax2)

    def load_data(self, prob_histogram_data):
        self.proba, self.targets = prob_histogram_data

        self.n_epochs = self.proba.shape[0]
        self._prepare_plot()
        return self

    def _prepare_plot(self):
        pass

    @staticmethod
    def update(i, ph, epoch_start=0):
        epoch = i + epoch_start

        correct = ((ph.proba[epoch] > .5) == ph.targets)
        tn = ph.proba[epoch, (ph.targets == 0) & correct]
        fn = ph.proba[epoch, (ph.targets == 0) & ~correct]
        tp = ph.proba[epoch, (ph.targets == 1) & correct]
        fp = ph.proba[epoch, (ph.targets == 1) & ~correct]

        for ax in (ph.ax1, ph.ax2):
            ax.clear()

        ph.ax1.set_title('{} - Epoch: {}'.format(ph.title[0], epoch + 1))
        ph.ax1.set_ylim([0, (ph.targets == 0).sum()])
        ph.ax1.set_xlabel('Probability')
        ph.ax1.set_ylabel('# of Cases')
        ph.ax1.hist(tn, bins=ph.bins, color='k', alpha=.4)
        ph.ax1.hist(fn, bins=ph.bins, color='r', alpha=.5)

        ph.ax2.set_title('{} - Epoch: {}'.format(ph.title[1], epoch + 1))
        ph.ax2.set_ylim([0, (ph.targets == 1).sum()])
        ph.ax2.set_xlabel('Probability')
        ph.ax2.set_ylabel('# of Cases')
        ph.ax2.hist(tp, bins=ph.bins, color='k', alpha=.4)
        ph.ax2.hist(fp, bins=ph.bins, color='r', alpha=.5)

        return ph.line

class LossAndMetric(Basic):
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
        self.loss, self.metric, self.metric_name = loss_and_metric_data
        self._title = '{} / Loss'.format(self.metric_name)

        self.n_epochs = self.loss.shape[0]
        self._prepare_plot()
        return self

    def _prepare_plot(self):
        self.ax.set_xlim([0, self.n_epochs])
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylim([0, 1.01 * self.metric.max()])
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
    def update(i, lm, epoch_start=0):
        epoch = i + epoch_start
        lm.ax.set_title('{} - Epoch: {}'.format(lm.title[0], epoch + 1))

        lm.line1.set_data(np.arange(0, epoch + 1), lm.metric[:epoch + 1])
        lm.line2.set_data(np.arange(0, epoch + 1), lm.loss[:epoch + 1])
        lm.point1.remove()
        lm.point1 = lm.ax.scatter(epoch, lm.metric[epoch], marker='o', color='k')
        lm.point2.remove()
        lm.point2 = lm.ax2.scatter(epoch, lm.loss[epoch], marker='o', color='r')

        return lm.line1, lm.line2

class LossHistogram(Basic):
    def __init__(self, ax):
        super(LossHistogram, self).__init__(ax)
        self.losses = None
        self._title = 'Losses'

    def __calc_scale(self, margin):
        loss_limits = np.array([self.losses.squeeze().min(), self.losses.squeeze().max()])
        loss_range = np.diff(loss_limits)[0]
        exponent = np.floor(np.log10(loss_range))
        magnitude = np.power(10, exponent)
        loss_limits = np.round(loss_limits + np.array([-margin, margin]) * magnitude, exponent.astype(np.int) + 1)
        intervals = (np.diff(loss_limits)[0] / magnitude + 1).astype(np.int)
        while 10 > intervals > 1:
            intervals = (intervals - 1) * 2 + 1
        loss_scale = np.linspace(loss_limits[0], loss_limits[1], intervals)
        return loss_scale

    def load_data(self, loss_hist_data):
        self.losses, = loss_hist_data
        self.bins = self.__calc_scale(margin=0)

        self.n_epochs = self.losses.shape[0]
        self._prepare_plot()
        return self

    def _prepare_plot(self):
        self.line = self.ax.plot([], [])

    @staticmethod
    def update(i, lh, epoch_start=0):
        epoch = i + epoch_start

        lh.ax.clear()

        lh.ax.set_title('{} - Epoch: {}'.format(lh.title[0], epoch + 1))
        lh.ax.set_ylim([0, lh.losses.shape[1]])
        lh.ax.set_xlabel('Loss')
        lh.ax.set_ylabel('# of Cases')
        lh.ax.hist(lh.losses.squeeze()[i], bins=lh.bins, color='k', alpha=.4)

        return lh.line