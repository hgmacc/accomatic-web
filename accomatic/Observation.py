"""
inherits from DataFiles (filePath, name, dataFrame, dateExtent)
"""
from accomatic.DataFileReader import DataFileReader

class Observation(DataFileReader):

    _type: str  # Individual results ultimately stored here

    def __init__(self, file_path=''):
        super().__init__(file_path, "Obs")

    @property
    def type(self) -> str:
        return self._type

    @property
    def missing_data_profile(self) -> str:
        # Report # of NAN in each col (one col = one site)

        return 'Not done yet'



