import numpy as np


def cross_entropy_loss(o, t):
    p = stable_softmax(o)
    log_likelihood = -np.log(p[t])
    return log_likelihood

def stable_softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)


def d_cross_entropy_loss(o, t):
    grad = stable_softmax(o)
    grad[t] -= 1
    return grad


def normalize_fr(hs):
    return (hs - hs.min(axis=0)) / (hs.max(axis=0) - hs.min(axis=0))


def d_tanh(x):
    return 1 / np.cosh(10 * np.tanh(x / 10))**2  # the tanh prevents oveflow


def h_seq_pre_processing(data):
    t_max = len(data)
    n_A = data[0][0].shape[0]
    n_B = data[0][1].shape[0]
    hs_A = np.zeros((t_max, n_A))
    hs_B = np.zeros((t_max, n_B))
    for t in range(t_max):
        hs_A[t] = data[t][0]
        hs_B[t] = data[t][1]

    return hs_A, hs_B


def sgn(x):
    return (x >= 0) * 2 - 1


def moving_average(data_set, periods=3):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, mode='valid')


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, label=None):
    ax = ax if ax is not None else plt.gca()
    if color is None:
        color = next(ax._get_lines.prop_cycler)['color']
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def angles_arrs(arr1, arr2):
    """
    args:
        arr1:[n,d]
        arr2:[n,d]
    returns:
        angles [n]
    """
    dprod = (arr1 * arr2).sum(axis=1)
    prodd = np.sqrt((arr1**2).sum(axis=1)) * np.sqrt((arr2**2).sum(axis=1))

    return 180 * np.arccos(np.clip(dprod / prodd, -1, 1)) / np.pi


def overlap_arrs(arr1, arr2):

    return (arr1 == arr2).sum(axis=1)


def sort_neurons_by_pos(fr_normalized):
    peak_fr_pos_list = np.argmax(fr_normalized, axis=0)
    return np.argsort(peak_fr_pos_list)