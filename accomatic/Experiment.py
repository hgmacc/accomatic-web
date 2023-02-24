import itertools
import os
import sys
from re import M
from typing import Dict, List

import pandas as pd
from accomatic.Ensemble import *
from accomatic.NcReader import *
from accomatic.Settings import *


class Experiment(Settings):
    _mod_dict: Dict
    _obs_dict: Dict
    _results: pd.DataFrame

    def __init__(self, sett_file_path="") -> None:
        super().__init__(sett_file_path)
        self.obs_dict = read_nc(self.obs_pth)
        self.mod_dict = read_geotop(self.model_pth)
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

    @mod_dict.setter
    def mod_dict(self, df) -> None:
        self._mod_dict = {
            site: Ensemble(
                site,
                df.loc[(df.index.get_level_values("sitename") == site)].droplevel(
                    "sitename"
                ),
            )
            for site in self.sites_list
        }

    @obs_dict.setter
    def obs_dict(self, df) -> None:
        self._obs_dict = {
            site: df.loc[(df.index.get_level_values("sitename") == site)].droplevel(
                "sitename"
            )
            for site in self.sites_list
        }

    @results.setter
    def results(self, df):
        a = list(itertools.product(self.sites_list, self.mod_names(), self.szn_list))
        sites = [x[0] for x in a]
        d = {
            "site": sites,
            "sim": [x[1] for x in a],
            "szn": [x[2] for x in a],
            "terr": [self.terr_dict()[x] for x in sites],
            "data_avail": np.ones(len(a)) * -999,
        }
        for acco in self._acco_list:
            d[acco] = np.ones(len(a)) * -999
        self._results = pd.DataFrame(data=d)

    def res_index(self, site, sim, szn):
        index = self._results.loc[
            (self._results["site"] == site)
            & (self._results["sim"] == sim)
            & (self._results["szn"] == szn)
        ]
        return index.index

    def mod(self, sitename="") -> pd.DataFrame:
        if sitename == "":
            return read_geotop(self._model_pth)
        else:
            return self._mod_dict[sitename].df

    def obs(self, sitename="") -> pd.DataFrame:
        if sitename == "":
            return read_nc(self._obs_pth)
        else:
            return self._obs_dict[sitename]

    def terr(self) -> List:
        return list(zip(self._terr_list, self._sites_list))

    def mod_names(self) -> pd.DataFrame:
        return next(iter(self.mod_dict.values())).model_list
