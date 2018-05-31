import itertools
import numpy as np

def load_data(n_dims=10, vertices=(-1., 1.), shuffle=True, seed=13):
    """

    Parameters
    ----------
    n_dims: int, optional
        Number of dimensions of the hypercube. Default is 10.
    edge: tuple of floats, optional
        Two vertices of an edge. Default is (-1., 1.).
    shuffle: boolean, optional
        If True, the points are shuffled. Default is True.
    seed: int, optional
        Random seed. Default is 13.

    Returns
    -------
    X, y: tuple of ndarray
        X is an array of shape (2 ** n_dims, n_dims) containing the
        vertices coordinates of the hypercube.
        y is an array of shape (2 ** n_dims, 1) containing the
        classes of the samples.
    """
    X = np.array(list(itertools.product(vertices, repeat=n_dims)))
    y = (np.sum(np.clip(X, a_min=0, a_max=1), axis=1) >= (n_dims / 2.0)).astype(np.int)

    # But we must not feed the network with neatly organized inputs...
    # so let's randomize them
    if shuffle:
        np.random.seed(seed)
        shuffled = np.random.permutation(range(X.shape[0]))
        X = X[shuffled]
        y = y[shuffled].reshape(-1, 1)

    return (X, y)
