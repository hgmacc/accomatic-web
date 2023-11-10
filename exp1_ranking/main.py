from accomatic.Experiment import Experiment
from accomatic.Stats import build


# s = time.time()

# # build_missing_data_pickles(n=1000, missing_amts=[0, 10, 25])
# # missing_boxplot(get_pickles("tests/test_data/pickles/n_10"))
# with open("20oct_100_exp.pickle", "rb") as f:
#     exp = pickle.load(f)

# # Get plot of
# terrain_timeseries(exp, "24OCT_terrain_timeseries.png")
# concatonate(exp)
# ranks_df = get_rank_distribution(exp)
# bs_heatmap(ranks_df, title="all")
# sys.exit()
# ranks_df = get_rank_distribution(exp)

# bs_heatmap(ranks_df[ranks_df.stat == "MAE"], title="mae_heat")
# bs_heatmap(ranks_df[ranks_df.stat == "WILL"], title="will_heat")
# bias_heatmap(ranks_df[ranks_df.stat == "BIAS"], title="bias_heat")
# bs_heatmap(ranks_df, title="all_heat")

# print(f"This took {format(time.time() - s, '.2f')}s to run.")

#### Create Experiment ##########################################

exp = Experiment("/home/hma000/accomatic-web/data/toml/run.toml")
build(exp)
