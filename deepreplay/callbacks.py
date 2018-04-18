from __future__ import division
import numpy as np
import h5py
from keras.callbacks import Callback

class ReplayData(Callback):
    def __init__(self, inputs, targets, filename, group_name):
        super(ReplayData, self).__init__()
        self.handler = h5py.File('{}'.format(filename), 'a')
        self.inputs = inputs
        self.targets = targets
        self.filename = filename
        self.group = None
        self.group_name = group_name
        self.current_epoch = -1
        return

    def _append_weights(self):
        for i, layer in enumerate(self.model.layers):
            layer_weights = layer.get_weights()
            for j, weights in enumerate(layer_weights):
                self.group['layer{}'.format(i)]['weights{}'.format(j)][self.current_epoch + 1] = weights

    def on_train_begin(self, logs={}):
        self.model.save('{}_model.h5'.format(self.group_name))
        self.n_epochs = self.params['epochs']

        self.group = self.handler.create_group(self.group_name)
        self.group.attrs['samples'] = self.params['samples']
        self.group.attrs['batch_size'] = self.params['batch_size']
        self.group.attrs['n_batches'] = np.ceil(self.params['samples'] / self.params['batch_size']).astype(np.int)
        self.group.attrs['n_epochs'] = self.n_epochs
        self.group.attrs['n_layers'] = len(self.model.layers)
        try:
            # Python 2
            self.group.attrs['activation_functions'] = [layer.activation.func_name
                                                        if hasattr(layer, 'activation')
                                                        else ''
                                                        for layer in self.model.layers]
        except AttributeError:
            # Python 3
            self.group.attrs['activation_functions'] = [np.string_(layer.activation.__name__)
                                                        if hasattr(layer, 'activation')
                                                        else np.string_('')
                                                        for layer in self.model.layers]
        self.group.create_dataset('inputs', data=self.inputs)
        self.group.create_dataset('targets', data=self.targets)

        self.group.create_dataset('loss', shape=(self.n_epochs,), dtype='f')
        for metric in self.model.metrics:
            self.group.create_dataset(metric, shape=(self.n_epochs,), dtype='f')

        for i, layer in enumerate(self.model.layers):
            layer_grp = self.group.create_group('layer{}'.format(i))
            layer_weights = layer.get_weights()
            for j, weights in enumerate(layer_weights):
                layer_grp.create_dataset('weights{}'.format(j),
                                         shape=(self.n_epochs + 1, ) + weights.shape,
                                         dtype='f')
        self._append_weights()
        return

    def on_train_end(self, logs={}):
        self.handler.close()
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.current_epoch = epoch
        return

    def on_epoch_end(self, epoch, logs={}):
        self._append_weights()
        self.group['loss'][epoch] = logs.get('loss')
        for metric in self.model.metrics:
            self.group[metric][epoch] = logs.get(metric, np.nan)
        return
