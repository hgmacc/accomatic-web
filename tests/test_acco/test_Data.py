from accomatic.Data import *
import pandas as pd
import glob


def test_data():
    obs: Observation = Observation("tests/test_data/test_obs_dir/test_obs.pickle")
    mod: List['Model'] = []
    for mod_file in glob.glob('tests/test_data/test_mod_dir/*.pickle'):
        mod.append(Model(mod_file))

    data: Data = Data(obs, mod)

    assert data.count == 2
    assert type(data.obs.df_dict['NGO-DD-1004_ST01'].index) == pd.DatetimeIndex
    # assert data.date_overlap['beg'] == pd.Timestamp("2016-06-01 00:00:00")
    # assert data.date_overlap['end'] == pd.Timestamp("2016-12-31 21:00:00")
    # assert [(mod.time_extent == data.obs.time_extent) for mod in data.models]
    pass
