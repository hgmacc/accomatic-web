import datetime
import os
import pickle
from typing import Dict, List

import pandas as pd

# Model = dictionary of 'point':'dataframe'
# obs = dictionary of 'point':'dataframe'


class DataFileReader:
    _file_path: str
    _name: str
    _df_dict: Dict[str, pd.DataFrame]
    _type: str
    _sites: List[str]

    def __init__(self, file_path="", type=""):
        if os.path.exists(file_path):
            self._file_path = file_path
            self._name = os.path.basename(file_path).split(".")[0]
        else:
            raise FileNotFoundError(
                "File path {} could not be found. Try again.".format(file_path)
            )

        self._df_dict = pd.read_pickle(file_path)
        self._type = type
        self._sites = self._df_dict.keys()

    @property
    def name(self) -> str:
        return self._name

    @property
    def df_dict(self) -> pd.DataFrame:
        return self._df_dict

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def type(self) -> str:
        return self._type

    @property
    def sites(self) -> List[str]:
        return self._sites

    @name.setter
    def name(self, n: str) -> None:
        if len(n) > 15:
            raise ValueError("This name is longer than 15 char. Try again.")
        else:
            self._name = n

    @df_dict.setter
    def df_dict(self, df_dict=Dict[str, pd.DataFrame]) -> None:
        self._df_dict = df_dict


"""
    @property
    def time_extent(self, site) -> Dict[str, pd.Timestamp]:
        beg = pd.Timestamp(self._df_dict[site].first_valid_index())
        end = pd.Timestamp(self._df_dict[site].last_valid_index())
        return {'beg': beg, 'end': end}
"""
