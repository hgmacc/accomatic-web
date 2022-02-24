import glob
from typing import List
import pandas as pd
from accomatic.Data import Data
from accomatic.Experiment import Experiment
from accomatic.Model import Model
from accomatic.Observation import Observation
from accomatic.Settings import Settings
from accomatic.Sites import Sites


def test_experiment():
    mod_list: List["Model"] = []
    for mod_file in glob.glob("tests/test_data/test_mod_dir/*.pickle"):
        print("success")
        mod_list.append(Model(mod_file))

    obs: Observation = Observation("tests/test_data/test_obs_dir/test_obs.pickle")
    sett: Settings = Settings("tests/test_data/test_toml_settings.toml")
    sites: Sites = Sites("tests/test_data/test_sites_list.csv")
    data: Data = Data(obs, mod_list)

    exp: Experiment = Experiment(data, sett, sites)

    pass
