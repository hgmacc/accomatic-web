from accomatic.Data import *
import pandas as pd

def test_data():
    obs: Observation = Observation("tests/test_data/test_obs_data.csv")
    mod: List['Model'] = [Model("tests/test_data/test_mod_data.csv")]
    data: Data = Data(obs, mod)
    assert data.count == 1
    assert type(data.obs.df.index) == pd.DatetimeIndex
    assert data.date_overlap['beg'] == pd.Timestamp("2016-06-01 00:00:00")
    assert data.date_overlap['end'] == pd.Timestamp("2016-12-31 21:00:00")
    assert [(mod.time_extent == data.obs.time_extent) for mod in data.models]
    pass
