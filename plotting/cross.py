def MAE_cross_plots(df):

    var = "rank"
    models = df.sim.unique().tolist()
    df["sett"] = df.szn + df.terr.astype(str)
    xlims, ylims = (df[var].min(), df[var].max()), (df[var].min(), df[var].max())
    df = df.set_index(["sim", "sett", "depth"])[var].unstack("depth")

    df50, df100 = df[[10, 50]].dropna(), df[[10, 100]].dropna()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10), squeeze=True)

    palette = ["#59473c", "#008080", "#F3700E", "#F50B00"]

    # XY SCATTER PLOT: 10 and 50
    plt.subplot(211)
    ax1.set_aspect("equal")

    for mod, col in zip(models, palette):
        s = f"{mod.upper()} $r$={np.corrcoef(df50.loc[mod][10],df50.loc[mod][50])[0][1]:.2f}"
        plt.scatter(df50.loc[mod][10], df50.loc[mod][50], s=5, c=col, label=s)

    plt.ylabel("MAE 0.5 m")
    plt.legend(fontsize="x-small")

    plt.plot(xlims, ylims, color="k", zorder=0)

    # XY SCATTER PLOT: 10 & 100
    plt.subplot(212)
    ax2.set_aspect("equal")

    for mod, col in zip(models, palette):
        s = f"{mod.upper()} $r$={np.corrcoef(df100.loc[mod][10],df100.loc[mod][100])[0][1]:.2f}"
        plt.scatter(df100.loc[mod][10], df100.loc[mod][100], s=5, c=col, label=s)

    # plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.xlabel("MAE 0.1 m")
    plt.ylabel("MAE 1.0 m")

    plt.plot(xlims, ylims, color="k", zorder=0)
    plt.legend(fontsize="x-small")

    fig.savefig(f"plane_plot_rank.png")

    fig.clf()
    plt.close(fig)
