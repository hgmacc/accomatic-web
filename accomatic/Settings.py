import sys
from typing import List

import toml


class Settings:
    _sett_file_path: str
    _model_pth: str
    _obs_pth: str
    _manifest: str
    _acco: bool
    _szn: bool
    _terr: bool
    _out_pth: str
    _output_plots: bool
    _term_summary: bool
    _acco_list: List[str]

    def __init__(self, sett_file_path=""):
        setting_toml = toml.load(sett_file_path)
        try:
            self._sett_file_path = sett_file_path
            self._model_pth = setting_toml["data"]["model_pth"]
            self._obs_pth = setting_toml["data"]["observations_pth"]
            self._manifest = setting_toml["data"]["manifest"]

            self._acco_list = setting_toml["experiment"]["acco_list"]
            self._acco = setting_toml["experiment"]["accordance"]
            self._szn = setting_toml["experiment"]["seasonal"]
            self._terr = setting_toml["experiment"]["terrain"]

            self._out_pth = setting_toml["output"]["out_pth"]
            self._output_plots = setting_toml["output"]["plots"]
            self._term_summary = setting_toml["output"]["terminal_summary"]

        except KeyError as e:
            print(
                f"Settings could not be configured due to {e} key error in TOML file."
            )
            sys.exit()

    @property
    def model_pth(self) -> bool:
        return self._model_pth

    @property
    def obs_pth(self) -> bool:
        return self._obs_pth
    
    @property
    def manifest(self) -> bool:
        return self._manifest

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
    def out(self) -> str:
        return self._out_pth
        
    @property
    def output_plots(self) -> bool:
        return self._output_plots

    @property
    def term_summary(self) -> bool:
        return self._term_summary
