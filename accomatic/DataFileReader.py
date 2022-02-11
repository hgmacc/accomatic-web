import pandas as pd
from typing import List
import datetime
import os


class DataFileReader:
    _file_path: str
    _name: str
    _df: pd.DataFrame  # Data contents
    _type: str

    def __init__(self, file_path='', type=''):
        if os.path.exists(file_path):
            self._file_path = file_path
            self._name = os.path.basename(file_path).split(".")[0]
        else:
            raise FileNotFoundError('File path {} could not be found. Try again.'.format(file_path))

        self._df = pd.read_csv(file_path, header=0, converters={'Date': pd.to_datetime}, dtype={'0.1': float},
                               index_col='Date')
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
    def type(self) -> str:
        return self._type

    @property
    def time_extent(self) -> dict:
        beg = pd.Timestamp(self._df.first_valid_index())
        end = pd.Timestamp(self._df.last_valid_index())
        return {'beg': beg, 'end': end}

    @name.setter
    def name(self, n: str) -> None:
        if len(n) > 15:
            raise ValueError("This name is longer than 15 char. Try again.")
        else:
            self._name = n

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        self._df = df
