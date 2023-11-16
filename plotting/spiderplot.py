import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

sys.path.append("../")
import accomatic
from accomatic.Stats import *
from accomatic.Experiment import *

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "10"


import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import datetime as date

import pickle


def radar_factory(num_vars, frame="circle"):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )
                return {"polar": spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def get_data(df):
    """
    This function takes the df.results from exp.results
    AFTER build() and concatonate() have been executed.
    """
    idx = pd.IndexSlice
    df = df.loc[idx[["res"], :, :, :]].droplevel("mode")

    for col in df.columns:
        m = list(df[col].values)
        m = [np.mean(cell.arr) for cell in m]
        df[col] = m

    rgrid = {
        "BIAS": {
            "min": df.loc[idx[:, :, "BIAS"]].min().min(),
            "max": df.loc[idx[:, :, "BIAS"]].max().max(),
        },
        "MAE": {
            "max": df.loc[idx[:, :, "MAE"]].max().max(),
        },
    }

    terrain_list = sorted(df.index.get_level_values("terr").unique().to_list())
    statistic_list = df.index.get_level_values("stat").unique()
    spider_list = [df.index.get_level_values("szn").unique().to_list()]
    simulation_list = df.columns

    df.rename(index=dict(zip(spider_list[0], list(range(1, 13)))), inplace=True)

    # ACCORDANCE ROWS: 1(acco) x n(terrains)
    for terrain in terrain_list:
        for statistic in statistic_list:
            data = []  # A list of lists; second part of spider_list tuple entry
            for simulation in simulation_list:
                data.append(
                    df.loc[idx[[terrain], :, [statistic]]][simulation]
                    .groupby("szn")
                    .mean()
                    .sort_index(level=["szn"])
                    .to_list()
                )
            spider_list.append((f"{statistic}+{terrain}", data))

    for statistic in statistic_list:
        stat_mean_data = []
        for simulation in simulation_list:
            # df_sim = (1) stat, (n) terrain, (1) model, (12) months
            stat_mean_data.append(
                df.loc[idx[:, :, [statistic]]][simulation]
                .groupby("szn")
                .mean()
                .sort_index(level=["szn"])
                .to_list()
            )
        spider_list.append((f"{statistic}+mean", stat_mean_data))

    return [spider_list, rgrid]


def spiderplot(df):

    data = get_data(df)
    rgrid = data[1]
    data = data[0]
    N = 12
    theta = radar_factory(N, frame="polygon")
    spoke_labels = ["JAN", "", "", "APR", "", "", "JUL", "", "", "OCT", "", ""]
    data.pop(0)

    rows = len(df.index.get_level_values("terr").unique()) + 1
    cols = len(df.index.get_level_values("stat").unique())

    fig, axs = plt.subplots(
        figsize=(cols * 4, rows * 4),
        ncols=cols,
        nrows=rows,
        subplot_kw=dict(projection="radar"),
    )
    fig.subplots_adjust(wspace=0.50, hspace=0.30, top=0.9, bottom=0.05)
    colors = ["#59473c", "#008080", "#f50b00", "#F3700E"]
    for ax, (title, case_data) in zip(axs.flat, data):
        ax.set_facecolor("white")

        if "MAE" in title:
            a = [int(round(i)) for i in np.linspace(0, rgrid["MAE"]["max"], 4)[1:3]]
            ax.set_ylim(0, rgrid["MAE"]["max"])
            ax.set_rgrids(a, zorder=10)

        if "WILL" in title:
            ax.set_ylim(0, 1)
            ax.set_rgrids([0.5], zorder=10)

        if "BIAS" in title:
            a = [
                int(round(i))
                for i in np.linspace(rgrid["BIAS"]["min"], rgrid["BIAS"]["max"], 4)[1:3]
            ]
            ax.set_ylim(rgrid["BIAS"]["min"], rgrid["BIAS"]["max"])
            ax.set_rgrids(a, zorder=10)

            zero = [0 for i in range(12)]
            ax.plot(theta, zero, color="k", linewidth=1)
            for d, color in zip(case_data, colors):
                # Plot fill
                d_pos = [bias if bias > 0 else 0 for bias in d]
                d_neg = [bias if bias <= 0 else 0 for bias in d]

                ax.fill_between(theta, zero, d_pos, facecolor="#ffbfbf")
                ax.fill_between(theta, d_neg, zero, facecolor="#bfbfff")

            for d, color in zip(case_data, colors):
                # Plot lines
                ax.plot(theta, d, color=color, linewidth=2)

        else:
            for d, color in zip(case_data, colors):
                ax.plot(theta, d, color=color)
                ax.fill(theta, d, facecolor=color, alpha=0.25, label="_nolegend_")

        ax.set_varlabels(spoke_labels)
    plt.savefig("out/spider.png")


"""
To run: 

exp = Experiment("/home/hma000/accomatic-web/data/toml/run.toml")
build(exp)
df = exp.results()
spiderplot(df) 

plot in: out/spider.png

"""
