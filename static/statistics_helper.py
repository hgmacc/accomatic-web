from operator import indexOf
import random
import seaborn as sns
import xarray as xr
import matplotlib
from matplotlib.dates import DateFormatter
from accomatic.NcReader import *
from sklearn.metrics import mean_absolute_error, mean_squared_error


class Cell:
    _arr: np.array

    def __init__(self):
        self._arr = []

    @property
    def arr(self) -> np.array:
        return np.array(self._arr)

    @arr.setter
    def arr(self, arr) -> None:
        self._arr = arr

    def ap(self, value: float) -> None:
        self._arr.append(value)

    def __repr__(self):
        return repr(self.arr)


class Data:
    _v: np.array

    def __init__(self, v):
        if type(v) == str:
            v = [int(i) for i in v.strip("[]").split(",")]
            v = np.array(v)
            rand_indices = np.random.randint(low=0, high=999, size=300)
            v[rand_indices] = np.nan
        self._v = v

    @property
    def v(self) -> np.array:
        return self._v

    @property
    def mean(self) -> float:
        return np.mean(self._v)

    @property
    def p(self) -> np.array:
        spl = make_interp_spline(range(10), self.v, k=3)
        return spl(np.linspace(-1, 1, 300))

    def __repr__(self):
        return repr(list(self.v))


def average_data(df_col):
    # df_col: column of np.arrays
    arr = np.array([i.v for i in df_col.to_list()])
    return Data(np.nanmean(arr, axis=0))


def rank_shifting_for_heatmap(df_col):
    return len(df_col.unique())


def std_dev(mod_ensemble):
    # From Luis (2020)
    M = len(mod_ensemble.columns)
    all_x_bars = []
    for model in mod_ensemble:
        all_x_bars.append(mod_ensemble[model].mean())
    x_bars_mean = np.mean(all_x_bars)
    return math.sqrt((((all_x_bars - x_bars_mean) ** 2).sum()) / (M - 1))


def variance(mod_ensemble):
    # From Luis (2020)
    M = len(mod_ensemble.columns)
    all_x_bars = []
    for model in mod_ensemble:
        all_x_bars.append(mod_ensemble[model].mean())
    x_bars_mean = np.mean(all_x_bars)
    return (((all_x_bars - x_bars_mean) ** 2).sum()) / (M - 1)


def willmott_refined_d(obs, mod):
    # From Luis (2020)
    o_mean = obs.mean()
    a = sum(abs(mod - obs))
    b = 2 * sum(abs(mod - o_mean))
    if a <= b:
        return 1 - (a / b)
    else:
        return (b / a) - 1


def r_score(obs, mod):
    return np.corrcoef(obs, mod)[0][1]


def nse_one(prediction, observation):
    o_mean = observation.mean()
    a = sum(abs(observation - prediction))
    b = sum(abs(observation - o_mean))
    return 1 - (a / b)


def bias(obs, mod):
    return np.mean(mod - obs)


def rmse(obs, mod):
    return mean_squared_error(obs, mod, squared=False)


acco_measures = {
    "RMSE": rmse,
    "R": r_score,
    "E1": nse_one,
    "MAE": mean_absolute_error,
    "WILL": willmott_refined_d,
    "BIAS": bias,
}

acco_rank = {
    "RMSE": "min",
    "R": "max",
    "E1": "max",
    "MAE": "min",
    "WILL": "max",
    "BIAS": "min",
}

time_code_months = {
    "ALL": list(range(1, 13)),
    "DJF": [1, 2, 12],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "FREEZE": [10, 11, 12, 1, 2, 3],
    "THAW": [4, 5, 6, 7, 8, 9],
    "JAN": [1],
    "FEB": [2],
    "MAR": [3],
    "APR": [4],
    "MAY": [5],
    "JUN": [6],
    "JUL": [7],
    "AUG": [8],
    "SEP": [9],
    "OCT": [10],
    "NOV": [11],
    "DEC": [12],
}


def heatmap(
    data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
