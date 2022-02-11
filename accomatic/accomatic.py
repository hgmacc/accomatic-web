import os
from accomatic.Model import Model
from accomatic.Sites import Sites
from accomatic.Settings import Settings
from accomatic.Data import Data
from accomatic.Observation import Observation
from accomatic.Experiment import Experiment
from typing import List


def accomatic(mod_dir, obs_csv, set_toml, sit_csv):

    # Collect model csvs
    mod_list: List['Model'] = []
    for mod_file in os.listdir(mod_dir):
        mod_file = os.path.join(mod_dir, mod_file)
        if os.path.isfile(mod_file):
            mod_list.append(Model(mod_file))

    # Initiate obs data, settings, sites
    obs: Observation = Observation(obs_csv)
    sett: Settings = Settings(set_toml)
    sites: Sites = Sites(sit_csv)
    data: Data = Data(obs, mod_list)

    exp: Experiment = Experiment(data, sett, sites)

    return True





