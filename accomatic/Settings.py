from typing import List

import toml
import sys


class Settings:
    _file_path: str
    _model_list: bool
    _model_pth: str
    _acco: bool
    _szn: bool
    _terr: bool
    _output_plots: bool
    _output_terminal_summary: bool
    _acco_list: List[str]

    def __init__(self, file_path=""):
        setting_toml = toml.load(file_path)
        try:
            self._file_path = file_path
            self._model_pth = setting_toml["data"]["model_pth"]
            self._obs_pth = setting_toml["data"]["observations_pth"]
            self._acco_list = setting_toml["experiment"]["acco_list"]
            self._acco = setting_toml["experiment"]["accordance"]
            self._szn = setting_toml["experiment"]["seasonal"]
            self._terr = setting_toml["experiment"]["terrain"]
            self._output_plots = setting_toml["output"]["plots"]
            self._output_terminal_summary = setting_toml["output"]["terminal_summary"]
            
        except KeyError as e:
            print(f"Settings could not be configured due to {e} key error in TOML file.")
            sys.exit()

    @property
    def model_pth(self) -> bool:
        return self._model_pth    \

    @property
    def obs_pth(self) -> bool:
        return self._obs_pth

    @property
    def acco(self) -> bool:
        return self._acco

    @property
    def szn(self) -> bool:
        return self._szn

    @property
    def terr(self) -> bool:
        return self._terr

    @property
    def acco_list(self) -> List[str]:
        return self._acco_list

    @property
    def output_plots(self) -> bool:
        return self._output_plots

    @property
    def output_terminal_summary(self) -> bool:
        return self._output_terminal_summary
