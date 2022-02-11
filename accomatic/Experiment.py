from accomatic.Sites import *
from accomatic.Settings import *
from accomatic.Data import *


class Experiment:
    _data: Data
    _sett: Settings
    _sites: Sites

    def __init__(self, data: Data, sett: Settings, sites: Sites):
        self._data = data
        self._sett = sett
        self._sites = sites

    @property
    def data(self) -> Data:
        return self._data

    @property
    def sett(self) -> Settings:
        return self._sett

    @property
    def sites(self) -> List['Sites']:
        return self._sites

    def run(self) -> None:
        if self.sett.exp_acco:
            # Runs accordance test
            pass
        if self.sett.exp_season:
            # Runs season test
            pass
        if self.sett.exp_terrain:
            # Runs terrain test
            pass





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