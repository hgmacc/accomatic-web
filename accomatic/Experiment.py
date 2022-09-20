from re import M
import sys
import pandas as pd
import os
from typing import List, Dict

import toml
from Settings import *
from NcReader import *
from accomatic.Ensemble import Ensemble


class Experiment(Settings):
    _mod_dict: Dict
    _obs: pd.DataFrame
    _results: pd.DataFrame

    def __init__(self, sett_file_path="") -> None:
        super().__init__(sett_file_path)
        self._obs = read_nc(self.obs_pth)
        self.mod_dict = read_geotop(self.model_pth)

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @property
    def mod_dict(self) -> Dict:
        return self._mod_dict

    @property
    def acco_list(self) -> List[str]:
        return self._acco_list

    @property
    def sites_list(self) -> List[str]:
        return self._sites_list

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

    def mod(self, sitename) -> pd.DataFrame:
        site_ens = self._mod_dict[sitename]
        return site_ens.df
