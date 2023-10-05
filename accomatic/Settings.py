import os
import re
import sys
from typing import Dict, List

import toml


class Settings:
    _model_pth: str
    _obs_pth: str
    _rank_csv_path: str
    _depth: float
    _boot_size: int
    _out_acco_pth: str
    _acco_list: List[str]
    _szn_list: List[str]
    _sites_list: List[str]
    _terr_list: List[str]
    _terr_desc: Dict[int, str]
    _missing_data: float

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

            self._rank_csv_path = setting_toml["data"]["rank_csv_path"]
            self._depth = setting_toml["data"]["depth"]
            self._boot_size = setting_toml["data"]["boot_size"]
            self._sites_list = setting_toml["data"]["sites_list"]
            self._missing_data = setting_toml["data"]["missing_data"]

            self._acco_list = setting_toml["experiment"]["acco_list"]
            self._szn_list = setting_toml["experiment"]["szn_list"]
            self._terr_list = setting_toml["experiment"]["terr_list"]

            terrain_descriptions = setting_toml["experiment"]["terr_desc"]
            self._terr_desc = dict(
                zip(
                    [i for i in range(1, len(terrain_descriptions) + 1)],
                    terrain_descriptions,
                )
            )

            if len(self._terr_list) != len(self.sites_list):
                print("ERROR: Terrains given in TOML file not equal to # of sites.")
                sys.exit()

            if len(self._terr_desc.values()) != len(set(self.terr_list)):
                print("WARNING: Check your terrain descriptions.")

        except KeyError as e:
            print(f"ERROR: Settings {e} key error in TOML file.")
            sys.exit()

    @property
    def model_pth(self) -> str:
        return self._model_pth

    @property
    def obs_pth(self) -> str:
        return self._obs_pth

    @property
    def rank_csv_path(self) -> str:
        return self._rank_csv_path

    @property
    def depth(self) -> str:
        return self._depth

    @property
    def boot_size(self) -> int:
        return self._boot_size

    @property
    def acco_list(self) -> List[str]:
        return self._acco_list

    @property
    def sites_list(self) -> List[str]:
        return self._sites_list

    @property
    def szn_list(self) -> List[str]:
        return self._szn_list

    @property
    def terr_list(self) -> List[str]:
        return self._terr_list

    @property
    def terr_desc(self) -> Dict[int, str]:
        return self._terr_desc

    @property
    def missing_data(self) -> float:
        return self._missing_data

    @missing_data.setter
    def missing_data(self, amt: int) -> None:
        self._missing_data = amt

    @boot_size.setter
    def boot_size(self, amt: int) -> None:
        self._boot_size = amt

    def terr_dict(self) -> Dict:
        return dict(zip(self._sites_list, self._terr_list))

    def __repr__(self):
        return (
            "\nExperiment setup: \n"
            + f" Model Path:\t\t{self.model_pth}\n"
            + f" Observations Path:\t{self.obs_pth}\n"
            + f" Acco Measures:\t\t{self.acco_list}"
        )
