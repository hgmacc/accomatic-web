import pandas as pd
from typing import List
import datetime
import os


class DataFileReader:

    file_path: str
    name: str
    df: pd.DataFrame  # Data contents
    type: str

    def __init__(self, file_path='', type=''):

        self.name = os.path.basename(file_path).split('.')[0]
        self.file_path = file_path
        self.df = pd.read_csv(file_path, index_col='Date')
        self.type = type

    def get_name(self) -> str:
        return self.name

    def get_df(self) -> pd.DataFrame:
        return self.df

    def get_file_path(self) -> str:
        return self.file_path

    def set_name(self, name: str) -> None:
        self.name = name

    def get_time_extent(self) -> List:
        # Test this to make sure valid index is actually datetime obj
        beg = self.df.first_valid_index()
        end = self.df.last_valid_index()
        return [beg, end]

