from typing import Dict

from accomatic.DataFileReader import DataFileReader


class Model(DataFileReader):
    _stats: Dict[str, float]  # Individual results ultimately stored here

    def __init__(self, file_path=""):
        super().__init__(file_path, "mod")
        self._stats = {}

    @property
    def stats(self) -> Dict[str, float]:
        return self._stats

    @stats.setter
    def stats(self, stat_dictionary: Dict[str, float]) -> None:
        self._stats = stat_dictionary
