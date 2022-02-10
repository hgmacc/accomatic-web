from accomatic.Observation import *


def test_observation():
    a = Observation("tests/test_data/test_csv_data.csv")
    assert a.name == "test_csv_data"
    assert a.file_path == "tests/test_data/test_csv_data.csv"
    assert a.type == "Obs"
    assert a.time_extent == ["2016-01-01 00:00:00", "2016-12-31 21:00:00"]
    pass
