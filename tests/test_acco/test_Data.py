from accomatic.Data import *
import pandas as pd

def test_data():
    obs: Observation = Observation("tests/test_data/test_csv_data.csv")
    mod: List['Model'] = [Model("tests/test_data/test_csv_overlap.csv")]
    data: Data = Data(obs, mod)

    assert data.count == 1
    assert data.date_overlap['beg'] == pd.Timestamp("2016-06-01 00:00:00")
    assert data.date_overlap['end'] == pd.Timestamp("2016-12-31 21:00:00")

    pass
