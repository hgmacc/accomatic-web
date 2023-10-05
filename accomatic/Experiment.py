import itertools
import os
import sys
from re import M
from typing import Dict, List, Union

import pandas as pd
from Ensemble import *
from accomatic.NcReader import *
from accomatic.Settings import *
from static.statistics_helper import average_data, time_code_months, Cell


class Experiment(Settings):
    _mod_dict: Dict
    _obs_dict: Dict
    _data: Dict
    _results: Union[Dict, pd.DataFrame]
    _old_results: pd.DataFrame

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

        self._results = {
            terr: {
                szn: {
                    "res": pd.DataFrame.from_dict(
                        {
                            stat: [Cell() for i in self.mod_names()]
                            for stat in self._acco_list
                        },
                        orient="index",
                        columns=self.mod_names(),
                    ),
                    "rank": pd.DataFrame.from_dict(
                        {
                            stat: [Cell() for i in self.mod_names()]
                            for stat in self._acco_list
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

        self.old_results = pd.DataFrame()

    @property
    def obs_dict(self) -> pd.DataFrame:
        return self._obs_dict

    @property
    def mod_dict(self) -> Dict:
        return self._mod_dict

    @property
    def old_results(self) -> pd.DataFrame:
        return self._old_results

    @property
    def data(self) -> Dict:
        return self._data

    def obs(self, sitename="") -> pd.DataFrame:
        if sitename == "":
            return self._obs
        else:
            return self._obs_dict[sitename]

    @data.setter
    def data(self, data):
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
    def results(self) -> Union[Dict, pd.DataFrame]:
        return self._results

    @results.setter
    def results(self, res: Union[Dict, pd.DataFrame]):
        self._results = res

    @old_results.setter
    def old_results(self, df):
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
        self._old_results = pd.DataFrame(data=d)

    def res_index(self, site, sim, szn):
        index = self._old_results.loc[
            (self._old_results["site"] == site)
            & (self._old_results["sim"] == sim)
            & (self._old_results["szn"] == szn)
        ]
        return index.index

    def res(self, sett=["sim", "szn", "terr"]) -> pd.DataFrame:
        # Arguably, the coolest function I've written.

        d1 = dict.fromkeys(["data_avail"], np.sum)
        d2 = dict.fromkeys(self._acco_list, average_data)
        d = {**d1, **d2}

        return self._old_results.groupby(sett, as_index=False).agg(d)

    def mod(self, sitename="") -> pd.DataFrame:
        if sitename == "":
            return self._mod
        else:
            return self._mod_dict[sitename].df

    def terr(self) -> List:
        return list(zip(self._terr_list, self._sites_list))

    def mod_names(self) -> pd.DataFrame:
        return next(iter(self.mod_dict.values())).model_list
