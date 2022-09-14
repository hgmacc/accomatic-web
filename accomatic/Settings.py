import sys
import os
from typing import List

import toml


class Settings:
    _model_pth: str
    _obs_pth: str
    _acco_list: List[str]
    _szn_list: List[str]

    def __init__(self, sett_file_path=""):
        setting_toml = toml.load(sett_file_path)
        try:
            path_error = "ERROR: Path '%s' does not exist."
            if os.path.exists(setting_toml["data"]["model_pth"]):
                self._model_pth = setting_toml["data"]["model_pth"]
            else: 
                print(path_error % setting_toml["data"]["model_pth"])
                sys.exit()
            
            if os.path.exists(setting_toml["data"]["observations_pth"]):
                self._obs_pth = setting_toml["data"]["observations_pth"]
            else: 
                print(path_error % setting_toml["data"]["observations_pth"])
                sys.exit()
            
            self._acco_list = setting_toml["experiment"]["acco_list"]

        except KeyError as e:
            print(f"ERROR: Settings {e} key error in TOML file.")
            sys.exit()

    @property
    def model_pth(self) -> bool:
        return self._model_pth

    @property
    def obs_pth(self) -> bool:
        return self._obs_pth
    
    @property 
    def acco_list(self) -> List[str]:
        return self._acco_list
    
    @property 
    def szn_list(self) -> List[str]:
        return self._szn_list

    def __repr__(self):
        return("Experiment setup: \n" +
                f" Model Path:\t\t{self.model_pth}\n" +
                f" Observations Path:\t{self.obs_pth}\n" +
                f" Acco Measures:\t\t{self.acco_list}")