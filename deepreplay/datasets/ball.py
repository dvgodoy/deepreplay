import numpy as np

def load_data(n_dims=10, radius=2.0, n_points=1000, only_sphere=False, shuffle=True, seed=13):
    """

    Parameters
    ----------
    n_dims: int, optional
        Number of dimensions of the n-ball. Default is 10.
    radius: float, optional
        Radius of the n-ball. Default is 2.0.
    n_points: int, optional
        Number of points in each parabola. Default is 1,000.
    only_sphere: boolean
        If True, generates a n-sphere, that is, a hollow n-ball.
        Default is False.
    shuffle: boolean, optional
        If True, the points are shuffled. Default is True.
    seed: int, optional
        Random seed. Default is 13.

    Returns
    -------
    X, y: tuple of ndarray
        X is an array of shape (n_points, n_dims) containing the
        points in the n-ball.
        y is an array of shape (n_points, 1) containing the
        classes of the samples.
    """
    points = np.random.normal(size=(1000, n_dims))
    sphere = points / np.linalg.norm(points, axis=1).reshape(-1, 1)
    if only_sphere:
        X = sphere
    else:
        X = radius * sphere * np.random.uniform(size=(n_points, 1))**(1 / n_dims)

    y = (np.abs(np.sum(X, axis=1)) > (radius / 2.0)).astype(np.int)

    # But we must not feed the network with neatly organized inputs...
    # so let's randomize them
    if shuffle:
        np.random.seed(seed)
        shuffled = np.random.permutation(range(X.shape[0]))
        X = X[shuffled]
        y = y[shuffled].reshape(-1, 1)

    return (X, y)
