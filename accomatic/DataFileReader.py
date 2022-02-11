import pandas as pd
from typing import List
import datetime
import os


class DataFileReader:
    file_path: str
    _name: str
    df: pd.DataFrame  # Data contents
    type: str

    def __init__(self, file_path="", type=""):
        if os.path.exists(file_path):
            self._file_path = file_path
            self._name = os.path.basename(file_path).split(".")[0]
        else:
            raise FileNotFoundError('File path {} could not be found. Try again.'.format(file_path))

        dtypes = {'Date': 'str', '0.1': 'float'}
        parse_dates = ['Date']
        self._df = pd.read_csv(file_path, index_col="Date", header=0, dtype=dtypes, parse_dates=parse_dates)
        self._type = type

    @property
    def name(self) -> str:
        return self._name

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def time_extent(self) -> dict:
        beg = pd.Timestamp(self._df.first_valid_index())
        end = pd.Timestamp(self._df.last_valid_index())
        return {'beg': beg, 'end': end}

    @name.setter
    def name(self, n: str) -> None:
        if len(n) > 5:
            raise ValueError("This name is longer than 5 char. Try again.")
        else:
            self._name = n
