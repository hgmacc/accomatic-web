def boot_vioplot(e, title=""):
    # site, stat, sim, label,

    stat = "MAE"
    if type(e) == Experiment():
        data = e.res(sett=["sim"])
        data = violin_helper_reorder_data(data, stat)

        label = data.sim.to_list()
        data_arr = np.array([i.v for i in data[stat].to_list()])
    else:
        data_arr = e
    fig, ax = plt.subplots(figsize=(len(data_arr) + 4, 8))

    bp = ax.violinplot(data_arr.T, showmeans=True)

    for patch, mod in zip(bp["bodies"], label):
        patch.set_facecolor(get_colour(mod))
        patch.set_alpha(1.0)

    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = bp[partname]
        vp.set_edgecolor("#000000")
        vp.set_linewidth(1)

    ax.set_ylabel(stat)
    ax.set_ylim(bottom=0, top=4)

    if title == "":
        import time

        title = f"vio_{time.time()}"
        print(title)

    legend_handles = [
        f"({a})" for i, a in zip(data.sim.to_list(), data["rank"].tolist())
    ]
    plt.legend(legend_handles, loc="lower left")
    plt.savefig(f"{os.path.dirname(os.path.realpath(__file__))}/{title}.png")
    plt.clf()
