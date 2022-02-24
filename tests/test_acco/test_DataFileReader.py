import pandas as pd
from accomatic.DataFileReader import *


def test_obs_DataFileReader():
    test_dfr = DataFileReader("tests/test_data/test_obs_dir/test_obs.pickle")
    test_dfr.name = "a_short_name"
    assert test_dfr.name != "test_obs"
    assert test_dfr.file_path == "tests/test_data/test_obs_dir/test_obs.pickle"
    assert test_dfr.time_extent['NGO-DD-1004_ST01'][0] == pd.Timestamp("2016-01-01 00:00:00")
    assert test_dfr.time_extent['NGO-DD-1004_ST01'][1] == pd.Timestamp("2016-12-31 23:00:00")
    assert type(test_dfr.df_dict["NGO-DD-1004_ST01"].index) == pd.DatetimeIndex
    pass


def test_mod_DataFileReader():
    test_dfr = DataFileReader("tests/test_data/test_mod_dir/test_erai_mod.pickle")
    test_dfr.name = "a_short_name"
    assert test_dfr.name != "test_erai_mod"
    assert test_dfr.file_path == "tests/test_data/test_mod_dir/test_erai_mod.pickle"
    assert type(test_dfr.df_dict["NGO-DD-1004_ST01"].index) == pd.DatetimeIndex
    pass

