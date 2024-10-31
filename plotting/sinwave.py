import matplotlib.patches as pat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline, make_interp_spline
import sys

sys.path.append("../")
from accomatic.Stats import d, d_1, d_r


plt.rcParams["font.size"] = "16"


def nse_stat(prediction, observation):
    o_mean = observation.mean()
    a = sum(abs(observation - prediction) ** 2)
    b = sum(abs(observation - o_mean) ** 2)
    return 1 - (a / b)


np.random.seed(1)
n = 1000

x = range(10)
xnew = np.linspace(-1, 1, 300)


plt.clf()


def build_df(x, datasets):

    r = ["{:.2f}".format(np.corrcoef(mod, sin(x))[0, 1]) for mod in datasets.values()]
    r2 = ["{:.2f}".format(float(i) ** 2) for i in r]
    mae = ["{:.2f}".format(np.mean(np.abs(mod - sin(x)))) for mod in datasets.values()]
    rmse = [
        "{:.2f}".format(np.sqrt(np.mean((mod - sin(x)) ** 2)))
        for mod in datasets.values()
    ]
    mae_outlier = [
        "{:.2f}".format(np.mean(np.abs(mod - datasets["real_outliers"])))
        for mod in datasets.values()
    ]
    nse = [
        "{:.2f}".format(nse_stat(mod, datasets["real_zero"]))
        for mod in datasets.values()
    ]
    sum_error = [
        "{:.2f}".format(np.sum(np.abs((mod - sin(x))))) for mod in datasets.values()
    ]
    sum_error_outlier = [
        "{:.2f}".format(np.sum(np.abs((mod - datasets["real_outliers"]))))
        for mod in datasets.values()
    ]
    rmse_outlier = [
        "{:.2f}".format(np.sqrt(np.mean((mod - datasets["real_outliers"]) ** 2)))
        for mod in datasets.values()
    ]
    bias = ["{:.2f}".format(np.mean(mod - sin(x))) for mod in datasets.values()]
    will = ["{:.2f}".format(d(mod, sin(x))) for mod in datasets.values()]
    will_1 = ["{:.2f}".format(d_1(mod, sin(x))) for mod in datasets.values()]
    will_r = ["{:.2f}".format(d_r(mod, sin(x))) for mod in datasets.values()]

    df = pd.DataFrame(
        columns=[
            "model",
            "r",
            "r2",
            "mae",
            "mae_outlier",
            "rmse",
            "sum_error_outlier",
            "rmse_outlier",
            "bias",
            "d",
            "d_1",
            "d_r",
            "sum_error",
            "nse",
        ]
    )

    df = pd.concat(
        [
            df,
            pd.DataFrame(
                {
                    "model": datasets.keys(),
                    "r": r,
                    "r2": r2,
                    "mae": mae,
                    "rmse": rmse,
                    "rmse_outlier": rmse_outlier,
                    "bias": bias,
                    "d": will,
                    "d_1": will_1,
                    "d_r": will_r,
                    "sum_error": sum_error,
                    "sum_error_outlier": sum_error_outlier,
                    "mae_outlier": mae_outlier,
                    "nse": nse,
                }
            ),
        ],
        ignore_index=True,
    )

    df = df.set_index(["model"]).T.astype(float)
    # print(df.to_latex())

    return df


def sin(
    x,
    freq=120,
    amp=10,
    lag=0,
    off=0,
    noise=False,
    noise_val=0.5,
    outliers=False,
    outliers_val=[10, 25],
):

    a = amp * np.sin((2.0 * np.pi / freq) * x + lag) + off
    if noise:
        noise_arr = (
            np.random.randint(low=noise_val * -10, high=noise_val * 10, size=len(x))
            / 10
        )
        tmp = []
        a = a + noise_arr * np.sin((2.0 * np.pi / 365 * 120) * x)
    if outliers and outliers_val == [10, 25]:
        for i in [10, 26, 30, 42, 50, 70, 72, 85, 90, 110]:
            a[i] = a[i] + (
                np.random.randint(low=-outliers_val[1], high=outliers_val[1], size=1)
            )
    if outliers and outliers_val != [10, 25]:
        for i in np.random.randint(low=0, high=outliers_val[0], size=120):
            a[i] = a[i] + (
                np.random.randint(low=-outliers_val[1], high=outliers_val[1], size=1)
            )
    return a


def test():
    # y = A*np.sin(omega*x+phase) + offset

    n = 3
    fig, ax = plt.subplots(n, figsize=(8, 5 * n))
    plt.tight_layout()
    x = np.array(range(120))

    datasets = {
        "pos_offset": sin(x, off=5),
        "neg_offset": sin(x, off=-5),
        "reduced_amp": sin(x, amp=2.5),
        "lag_0": sin(x, lag=1.5, amp=1) * -0.01,
        "reduced_amp_lag": sin(x, amp=1, lag=0.5),
        "offset = 0.5": sin(x, off=0.5),
        "step_wise": np.around(sin(x) * 1 / 6) / (1 / 6),
        "low_corr": sin(x, outliers=True, outliers_val=[40, 10]),
        "real": sin(x, noise=True),
        "reduced_amp_real": sin(x, amp=1, noise=True, noise_val=0.1),
        "real_outliers": sin(x, noise=True, outliers=True),
        "reduced_amp_perfect": sin(x, amp=1),
        "real_zero": sin(x, lag=1.5, amp=1) * 0.01
        + np.random.randint(low=-10, high=10, size=120) / 10000,
    }

    datasets["low_rmse"] = sin(x) + 2  # np.copy(datasets["real_outliers"]) + 2
    datasets["sin_wave"] = sin(x)  # np.copy(datasets["real_outliers"])
    datasets["no_snow"] = np.concatenate((sin(x)[:60] + 0.05, sin(x)[60:] * 2))
    datasets["high_var"] = sin(x, noise=True, noise_val=5)

    for i in [10, 26, 30, 42, 50, 70, 72, 85, 90, 110]:
        datasets["low_rmse"][i] = datasets["real_outliers"][i]
        datasets["sin_wave"][i] = datasets["sin_wave"][i - 1]
        cir = pat.Ellipse(
            (i, datasets["real_outliers"][i]),
            width=2.5,
            height=1.6,
            alpha=0.5,
            color="#8fbdbc",
            fill=True,
        )
        ax[0].add_patch(cir)

    # AMPLITUDE
    ax[0].plot(x, datasets["real_outliers"], "k", linewidth=3, label="Observations")
    ax[0].plot(x, datasets["sin_wave"], c="#f4c18e", linewidth=1.5, label="Model 1")
    ax[0].plot(x, datasets["low_rmse"], c="#f28e89", linewidth=1.5, label="Model 2")

    # BIAS
    ax[1].plot(x, datasets["real"], "k", linewidth=3, zorder=5)
    ax[1].plot(x, datasets["pos_offset"], c="#f4c18e")
    ax[1].plot(x, datasets["neg_offset"], c="#f28e89")
    ax[1].plot(x, datasets["step_wise"], c="#8fbdbc")

    # CORRELATION
    ax[2].plot(
        x, datasets["reduced_amp_real"], "k", linewidth=3, label="_Real_", zorder=-1
    )
    ax[2].plot(x, datasets["reduced_amp_lag"], c="#f28e89", label="lag", zorder=5)
    ax[2].plot(x, datasets["reduced_amp"], c="#f4c18e")
    ax[2].plot(x, datasets["reduced_amp_perfect"], c="#8fbdbc")

    df = build_df(x, datasets)

    ax0_df = df.loc[["mae_outlier", "rmse_outlier", "sum_error_outlier"]][
        ["sin_wave", "low_rmse"]
    ]
    ax0_table = ax[0].table(
        cellText=(ax0_df.to_numpy()),
        colWidths=[0.1] * 3,
        rowLabels=["$MAE$", "$RMSE$", "Total Error"],
        colLabels=["Model 1", "Model 2"],
        colColours=["#f4c18e", "#f28e89"],
        loc="upper right",
    )
    ax0_table.set_fontsize(18)
    ax0_table.scale(1, 1.5)

    ax1_df = df.loc[["mae", "r", "bias"]][["pos_offset", "neg_offset", "step_wise"]]
    ax1_table = ax[1].table(
        cellText=ax1_df.to_numpy(),
        colWidths=[0.1] * 3,
        rowLabels=["$MAE$", "$r$", "$MBE$"],
        colLabels=["Model 1", "Model 2", "Model 3"],
        colColours=["#f4c18e", "#f28e89", "#8fbdbc"],
        loc="upper right",
    )
    ax1_table.set_fontsize(20)
    ax1_table.scale(1, 1.7)

    ax2_df = df.loc[["r", "r2", "d_r"]][
        ["reduced_amp", "reduced_amp_lag", "reduced_amp_perfect"]
    ]
    ax2_table = ax[2].table(
        cellText=ax2_df.to_numpy(),
        colWidths=[0.1] * 3,
        rowLabels=["$r$", "$r^2$", "$d_r$"],
        colLabels=["Model 1", "Model 2", "Model 3"],
        colColours=["#f4c18e", "#f28e89", "#8fbdbc"],
        loc="upper right",
    )
    ax2_table.set_fontsize(18)
    ax2_table.scale(1, 1.7)

    alpha = ["A", "B", "C", "D", "E"]
    for i, l in zip(range(n), alpha[:n]):
        if l == "B":
            ax[i].text(1, 15, l, fontsize=20, fontweight="bold")
        elif l == "C":
            ax[i].text(1, 2, l, fontsize=20, fontweight="bold")
        else:
            ax[i].text(1, 22, l, fontsize=20, fontweight="bold")
        ax[i].set_yticks([i for i in range(-20, 30, 10)])
        ax[i].set_xticks([])
    ax[n - 1].set_xticks(
        [i * 10 for i in range(13)], labels=[str(i) for i in range(13)]
    )

    ax[2].set_ylim(-3, 3)
    ax[i].set_yticks([-3, -1.5, 0, 1.5, 3])

    plt.savefig("/home/hma000/accomatic-web/plotting/out/examples/sinwave.png")


if __name__ == "__main__":
    test()
