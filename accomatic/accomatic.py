import glob
import os
from typing import List

from accomatic.Data import Data
from accomatic.Experiment import Experiment
from accomatic.Model import Model
from accomatic.Observation import Observation
from accomatic.Settings import Settings
from accomatic.Sites import Sites

# ASSUME PICKLED DICTIONARIES
# { str = pd.DataFrame }


def accomatic(model_dir, obs_pth, settings_toml, sites_csv):

    # Collect model csvs
    mod_list: List["Model"] = []
    for mod_file in glob.glob(model_dir + "/*.pickle"):
        print("success")
        mod_list.append(Model(mod_file))

    # Initiate obs data, settings, sites
    obs: Observation = Observation(obs_pth)
    sett: Settings = Settings(settings_toml)
    sites: Sites = Sites(sites_csv)
    data: Data = Data(obs, mod_list)

    exp: Experiment = Experiment(data, sett, sites)
    # exp.run()

    return True
