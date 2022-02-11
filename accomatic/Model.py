from accomatic.DataFileReader import DataFileReader


class Model(DataFileReader):
    _stats: dict  # Individual results ultimately stored here

    def __init__(self, file_path=""):
        super().__init__(file_path, 'mod')
        self._stats = {}

    @property
    def stats(self) -> dict:
        return self._stats

    @stats.setter
    def stats(self, stat_dictionary: dict) -> None:
        self._stats = stat_dictionary
