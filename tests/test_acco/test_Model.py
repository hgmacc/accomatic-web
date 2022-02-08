from accomatic.Model import *


def test_Model():
    a = Model('tests/test_data/NGO-DD-1004_ST01.csv')
    assert a.get_name() == 'NGO-DD-1004_ST01'
    assert a.get_file_path() == 'tests/test_data/NGO-DD-1004_ST01.csv'
    assert bool(a.get_stats()) == False
    assert a.get_time_extent() == ['2016-01-01 00:00:00', '2016-12-31 21:00:00']
    pass

