from typing import List
import toml

class Settings:
    file_path: str
    pickled_data: bool
    exp_acco: bool
    exp_season: bool
    exp_terrain: bool
    output_plots: bool
    output_terminal_summary: bool
    exp_time_extent: List[str]
    acco_list: List[str]

    def __init__(self, file_path=''):
        setting_toml = toml.load(file_path)

        self.file_path = file_path
        self.pickled_data = setting_toml['data']['pickled']
        self.exp_acco = setting_toml['experiment']['accordance_measure']
        self.exp_season = setting_toml['experiment']['seasonal']
        self.exp_terrain = setting_toml['experiment']['terrain']
        self.output_plots = setting_toml['results']['output_plots']
        self.output_terminal_summary = setting_toml['results']['output_terminal_summary']
        self.exp_time_extent = [setting_toml['experiment']['beg'], setting_toml['experiment']['end']]
        self.acco_list = setting_toml['experiment']['accordance']


    def get_pickled_data(self) -> bool:
        return self.pickled_data

    def get_exp_acco(self) -> bool:
        return self.exp_acco

    def get_exp_season(self) -> bool:
        return self.exp_season

    def get_exp_terrain(self) -> bool:
        return self.exp_terrain

    def get_exp_time_extent(self) -> List[str]:
        return self.exp_time_extent

    def get_acco_list(self) -> List[str]:
        return self.acco_list

    def get_output_plots(self) -> bool:
        return self.output_plots

    def get_output_terminal_summary(self) -> bool:
        return self.output_terminal_summary




