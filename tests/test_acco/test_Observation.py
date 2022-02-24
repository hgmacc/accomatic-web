import pandas as pd
from accomatic.Observation import *


def test_observation():
    a = Observation("tests/test_data/test_obs_dir/test_obs.pickle")
    assert a.name == "test_obs"
    assert a.file_path == "tests/test_data/test_obs_dir/test_obs.pickle"
    assert a.type == "obs"
    assert len(a.sites) == 5
    assert type(a.df_dict["NGO-DD-1004_ST01"].index) == pd.DatetimeIndex
    assert a.time_extent['NGO-DD-1004_ST01'][0] == pd.Timestamp("2016-01-01 00:00:00")
    assert a.time_extent['NGO-DD-1004_ST01'][1] == pd.Timestamp("2016-12-31 23:00:00")
    assert a.missing_data['NGO-DD-1004_ST01'] == 0.0
    pass
