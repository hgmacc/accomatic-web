import sys
import pandas as pd
import os
from typing import List

import toml
from Settings import *
from NcReader import *

class Experiment(Settings):
    _mod: pd.DataFrame
    _obs: pd.DataFrame
    _results: pd.DataFrame

    def __init__(self, sett_file_path = "") -> None:
        super().__init__(sett_file_path)
        self._obs, self._mod = getdf(self.model_pth, self.obs_pth)

    @property
    def obs(self) -> pd.DataFrame:
        return self._obs

    @property
    def mod(self) -> pd.DataFrame:
        return self._mod
    
    @property 
    def acco_list(self) -> List[str]:
        return self._acco_list

    @property
    def nmod(self) -> int:
        # return number of models in model nc file
        # i.e. Geotop x 3 reanal = 3 models
        return 0
    
    @property 
    def szn_list(self) -> List[str]:
        # Based on toml, which seasons are being considered?
        return [0]
    
    def __repr__(self):
        return("Experiment setup: \n" +
                f" Model Path:\t\t{self.model_pth}\n" +
                f" Observations Path:\t{self.obs_pth}\n" +
                f" Acco Measures:\t\t{self.acco_list}")