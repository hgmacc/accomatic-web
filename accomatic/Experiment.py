"""

    # holds models
    # and outputs
    # can generate output dictionaries and put stats into models

for m in all_accordance_funcs:
    model_means[m] = all_accordance_results[m].mean()
    # print("Means:\n",all_accordance_results[m].mean())
    if m in ['RMSE', 'MAE']:
        model_ranks[m] = all_accordance_results[m].mean().rank(axis=0, method='average', ascending=True)
    else:
        model_ranks[m] = all_accordance_results[m].mean().rank(axis=0, method='average', ascending=False)
    accordance_max[m] = all_accordance_results[m].max()
    accordance_min[m] = all_accordance_results[m].min()

"""
from accomatic.Sites import *


class Experiment(Data, Sites, Settings):

    Data(models + obs)
    Sites
    Settings
