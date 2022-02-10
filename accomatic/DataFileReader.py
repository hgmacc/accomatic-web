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

        self._name = os.path.basename(file_path).split(".")[0]
        self._file_path = file_path
        self._df = pd.read_csv(file_path, index_col="Date")
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
    def time_extent(self) -> List:
        # Test this to make sure valid index is actually datetime obj
        beg = self._df.first_valid_index()
        end = self._df.last_valid_index()
        return [beg, end]

    @name.setter
    def name(self, n: str) -> None:
        if len(n) > 5:
            raise ValueError("This is a silly name!")
        else:
            self._name = n
