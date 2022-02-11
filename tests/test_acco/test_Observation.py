from accomatic.Observation import *
import pandas as pd

def test_observation():
    a = Observation("tests/test_data/test_csv_data.csv")
    assert a.name == "test_csv_data"
    assert a.file_path == "tests/test_data/test_csv_data.csv"
    assert a.type == "obs"
    assert type(a.df.index) == pd.DatetimeIndex
    assert a.time_extent['beg'] == pd.Timestamp("2016-01-01 00:00:00")
    assert a.time_extent['end'] == pd.Timestamp("2016-12-31 21:00:00")
    assert a.missing_data == 0.06
    pass
