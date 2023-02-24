from typing import List

import pandas as pd
from accomatic.NcReader import *


class Ensemble:
    _sitename: str
    _df: pd.DataFrame
    _model_list: List[str]

    def __init__(self, sitename="", df=pd.DataFrame()) -> None:
        self._sitename = sitename
        self._df = df
        self._model_list = self._df.columns.values.tolist()

    @property
    def sitename(self) -> str:
        return self._sitename

    @property
    def model_list(self) -> List[str]:
        return self._model_list

    @property
    def df(self) -> pd.DataFrame:
        return self._df
