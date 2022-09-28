from re import M
import sys
import pandas as pd
import os
from typing import List, Dict

import toml
from Settings import *
from NcReader import *
from Ensemble import *


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
    def results(self, df) -> None:
        iterables = list(self.sites_list), list(self.szn_list)
        index = pd.MultiIndex.from_product(iterables, names=["site", "season"])
        df = pd.DataFrame(columns=self.acco_list, index=index)
        self._results = df

    def mod(self, sitename) -> pd.DataFrame:
        return self._mod_dict[sitename].df

    def obs(self, sitename) -> pd.DataFrame:
        return self._obs_dict[sitename]

    def terr(self) -> List:
        return list(zip(self._terr_list, self._sites_list))
