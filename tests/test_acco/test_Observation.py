from accomatic.Observation import *
import pandas as pd

def test_observation():
    a = Observation("tests/test_data/test_obs_dir/test_obs.pickle")
    assert a.name == "test_obs"
    assert a.file_path == "tests/test_data/test_obs_dir/test_obs.pickle"
    assert a.type == "obs"
    assert len(a.sites) == 5
    assert type(a.df_dict['NGO-DD-1004_ST01'].index) == pd.DatetimeIndex
    # assert a.missing_data['NGO-DD-1004_ST01'] == 0.06
    pass


