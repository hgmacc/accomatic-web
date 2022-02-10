from accomatic.DataFileReader import *


def test_DataFileReader():
    a = DataFileReader("tests/test_data/test_csv_data.csv")
    a.name = "aNam"
    assert a.name != "test_csv_data"
    assert a.file_path == "tests/test_data/test_csv_data.csv"
    assert a.time_extent == ["2016-01-01 00:00:00", "2016-12-31 21:00:00"]
    pass
