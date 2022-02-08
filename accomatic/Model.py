from accomatic.DataFileReader import DataFileReader

class Model(DataFileReader):

    stats: dict  # Individual results ultimately stored here
    type: str    # Differentiate between model and obs

    def __init__(self, file_path=''):
        super().__init__(file_path)
        self.stats = {}
        self.type = 'Model'

    def get_stats(self) -> dict:
        return self.stats

    def set_stats(self, stat_dictionary: dict) -> None:
        self.stats = stat_dictionary

    def get_type(self) -> str:
        return self.type
