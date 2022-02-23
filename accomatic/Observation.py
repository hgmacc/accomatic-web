from accomatic.DataFileReader import DataFileReader
from typing import Dict


class Observation(DataFileReader):
    _missing_data: Dict[str, float]

    def __init__(self, file_path=""):
        super().__init__(file_path, 'obs')
        self._missing_data = {}
        for site in self.df_dict:
            df = self.df_dict[site]
            percent = df.iloc[:, 0].isna().sum() / len(df.index) * 100
            self._missing_data[site] = round(percent, 2)

    @property
    def missing_data(self) -> Dict[str, float]:
        return self._missing_data

