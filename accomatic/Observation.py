from accomatic.DataFileReader import DataFileReader


class Observation(DataFileReader):

    def __init__(self, file_path=""):
        super().__init__(file_path, 'obs')

    @property
    def missing_data(self) -> float:
        percent = self.df.iloc[:, 0].isna().sum() / len(self.df.index) * 100
        return round(percent, 2)

