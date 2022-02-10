from typing import List
import toml

class Settings:
    _file_path: str
    _pickled_data: bool
    _exp_acco: bool
    _exp_season: bool
    _exp_terrain: bool
    _output_plots: bool
    _output_terminal_summary: bool
    _exp_time_extent: List[str]
    _acco_list: List[str]

    def __init__(self, file_path=''):
        setting_toml = toml.load(file_path)

        self._file_path = file_path
        self._pickled_data = setting_toml['data']['pickled']
        self._exp_acco = setting_toml['experiment']['accordance_measure']
        self._exp_season = setting_toml['experiment']['seasonal']
        self._exp_terrain = setting_toml['experiment']['terrain']
        self._output_plots = setting_toml['results']['output_plots']
        self._output_terminal_summary = setting_toml['results']['output_terminal_summary']
        self._exp_time_extent = [setting_toml['experiment']['beg'], setting_toml['experiment']['end']]
        self._acco_list = setting_toml['experiment']['accordance']

    @property
    def pickled_data(self) -> bool:
        return self._pickled_data

    @property
    def exp_acco(self) -> bool:
        return self._exp_acco

    @property
    def exp_season(self) -> bool:
        return self._exp_season

    @property
    def exp_terrain(self) -> bool:
        return self._exp_terrain

    @property
    def exp_time_extent(self) -> List[str]:
        return self._exp_time_extent

    @property
    def acco_list(self) -> List[str]:
        return self._acco_list

    @property
    def output_plots(self) -> bool:
        return self._output_plots

    @property
    def output_terminal_summary(self) -> bool:
        return self._output_terminal_summary




