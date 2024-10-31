import itertools
import os
import sys
from re import M
from typing import Dict, List, Union

import pandas as pd
from accomatic.Ensemble import *
from accomatic.NcReader import *
from accomatic.Settings import *
from accomatic.Stats import average_data, time_code_months, Cell


def read_exp(file_path=""):
    with open(file_path, "rb") as f_gst:
        exp = pickle.load(f_gst)
        return exp


class Experiment(Settings):
    _mod_dict: Dict
    _obs_dict: Dict
    _data: Dict
    _results: Union[Dict, pd.DataFrame]
    _rank_dist: pd.DataFrame
    _bias_dist: pd.DataFrame

    def __init__(self, sett_file_path="") -> None:
        super().__init__(sett_file_path)

        self._obs = read_nc(
            self._obs_pth,
            sitename=self.sites_list,
            depth=self.depth,
        )

        self._obs_dict = {
            site: self._obs.loc[
                (self._obs.index.get_level_values("sitename") == site)
            ].droplevel("sitename")
            for site in self.sites_list
        }

        self._mod = read_geotop(
            file_path=self._model_pth, depth=self.depth, sitename=self.sites_list
        )
        self._mod_dict = {
            site: Ensemble(
                site,
                self._mod.loc[
                    (self._mod.index.get_level_values("sitename") == site)
                ].droplevel("sitename"),
            )
            for site in self.sites_list
        }

        self._results = {
            terr: {
                szn: {
                    "res": pd.DataFrame.from_dict(
                        {
                            stat: [Cell() for i in self.mod_names()]
                            for stat in self._stat_list
                        },
                        orient="index",
                        columns=self.mod_names(),
                    ),
                    "rank": pd.DataFrame.from_dict(
                        {
                            stat: [Cell() for i in self.mod_names()]
                            for stat in self._stat_list
                        },
                        orient="index",
                        columns=self.mod_names(),
                    ),
                }
                for szn in self.szn_list
            }
            for terr in self.terr_list
        }

        self.data = {
            terr: {szn: [] for szn in self.szn_list} for terr in self.terr_list
        }

        self._rank_dist = pd.DataFrame()
        self._bias_dist = pd.DataFrame()

    @property
    def obs_dict(self) -> pd.DataFrame:
        return self._obs_dict

    @property
    def mod_dict(self) -> Dict:
        return self._mod_dict

    @property
    def data(self) -> Dict:
        return self._data

    def obs(self, sitename="") -> pd.DataFrame:
        if sitename == "":
            return self._obs
        else:
            return self._obs_dict[sitename]

    @data.setter
    def data(self, data) -> None:
        for terr_i in set(self.terr_list):
            terrain_sites = [
                site for site, terr in self.terr_dict().items() if terr == terr_i
            ]
            for site in terrain_sites:
                df = self.obs(site).join(self.mod(site)).dropna()
                df.index = pd.to_datetime(df.index)
                df.index.names = ["Date"]
                for szn_i in self.szn_list:
                    df_szn = df[df.index.month.isin(time_code_months[szn_i])]
                    data[terr_i][szn_i].extend(
                        [
                            df_szn[df_szn.index.year.isin([year_i])]
                            for year_i in df_szn.index.year.unique()
                            if len(df_szn[df_szn.index.year.isin([year_i])]) > 27
                        ]
                    )
        self._data = data

    @property
    def bias_dist(self) -> pd.DataFrame:
        return self._bias_dist

    @bias_dist.setter
    def bias_dist(self, df: pd.DataFrame) -> None:
        self._bias_dist = df

    @property
    def rank_dist(self) -> pd.DataFrame:
        return self._rank_dist

    @rank_dist.setter
    def rank_dist(self, df: pd.DataFrame) -> None:
        self._rank_dist = df

    @property
    def results(self) -> Union[Dict, pd.DataFrame]:
        return self._results

    @results.setter
    def results(self, res: Union[Dict, pd.DataFrame]) -> None:
        self._results = res

    def mod(self, sitename="") -> pd.DataFrame:
        if sitename == "":
            return self._mod
        else:
            return self._mod_dict[sitename].df

    def terr(self) -> List:
        return list(zip(self._terr_list, self._sites_list))

    def mod_names(self) -> pd.DataFrame:
        return next(iter(self.mod_dict.values())).model_list
