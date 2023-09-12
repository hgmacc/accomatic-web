import itertools
import os
import sys
from re import M
from typing import Dict, List

import pandas as pd
from Ensemble import *
from accomatic.NcReader import *
from accomatic.Settings import *
from static.statistics_helper import average_data


class Experiment(Settings):
    _mod_dict: Dict
    _obs_dict: Dict
    _results: pd.DataFrame

    def __init__(self, sett_file_path="") -> None:
        super().__init__(sett_file_path)

        self._obs = read_nc(self._obs_pth, sitename=self.sites_list, depth=self.depth)
        self._obs_dict = {
            site: self._obs.loc[
                (self._obs.index.get_level_values("sitename") == site)
            ].droplevel("sitename")
            for site in self.sites_list
        }

        self._mod = read_geotop(file_path=self._model_pth, sitename=self.sites_list)
        self._mod_dict = {
            site: Ensemble(
                site,
                self._mod.loc[
                    (self._mod.index.get_level_values("sitename") == site)
                ].droplevel("sitename"),
            )
            for site in self.sites_list
        }

        self.results = pd.DataFrame()

    @property
    def obs_dict(self) -> pd.DataFrame:
        return self._obs_dict

    @property
    def mod_dict(self) -> Dict:
        return self._mod_dict

    @property
    def results(self) -> pd.DataFrame:
        return self._results

    @results.setter
    def results(self, df):
        a = list(itertools.product(self.sites_list, self.mod_names(), self.szn_list))
        sites = [x[0] for x in a]
        d = {
            "site": sites,
            "sim": [x[1] for x in a],
            "szn": [x[2] for x in a],
            "terr": [self.terr_dict()[x] for x in sites],
            "data_avail": np.full((len(a),), np.nan),
        }
        for acco in self._acco_list:
            d[acco] = np.full((len(a),), np.nan)
        self._results = pd.DataFrame(data=d)

    def res_index(self, site, sim, szn):
        index = self._results.loc[
            (self._results["site"] == site)
            & (self._results["sim"] == sim)
            & (self._results["szn"] == szn)
        ]
        return index.index

    def res(self, sett=["sim", "szn", "terr"]) -> pd.DataFrame:
        # Arguably, the coolest function I've written.

        d1 = dict.fromkeys(["data_avail"], np.sum)
        d2 = dict.fromkeys(self._acco_list, average_data)
        d = {**d1, **d2}

        return self._results.groupby(sett, as_index=False).agg(d)

    def mod(self, sitename="") -> pd.DataFrame:
        if sitename == "":
            return self._mod
        else:
            return self._mod_dict[sitename].df

    def obs(self, sitename="") -> pd.DataFrame:
        if sitename == "":
            return self._obs
        else:
            return self._obs_dict[sitename]

    def terr(self) -> List:
        return list(zip(self._terr_list, self._sites_list))

    def mod_names(self) -> pd.DataFrame:
        return next(iter(self.mod_dict.values())).model_list
