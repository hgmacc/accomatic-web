import glob
import os
import re
from typing import List

import pandas as pd
from accomatic.Model import *


def test_Model():
    mod_list: List["Model"] = []

    for mod_file in glob.glob("tests/test_data/test_mod_dir/*.pickle"):
        mod_list.append(Model(mod_file))

    for a in mod_list:
        assert bool(re.search("test_.*_mod", a.name))
        assert len(a.sites) == 5
        assert bool(
            re.search("tests/test_data/test_mod_dir/test_.*_mod.pickle", a.file_path)
        )
        assert bool(a.stats) == False
        assert type(a.df_dict["NGO-DD-1004_ST01"].index) == pd.DatetimeIndex
        assert a.stats == {}
    pass
