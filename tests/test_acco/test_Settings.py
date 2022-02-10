from accomatic.Settings import *


def test_settings():
    a = Settings('tests/test_data/test_toml_settings.toml')
    assert a.pickled_data == True
    assert a.exp_season == True
    assert a.exp_acco == True
    assert a.exp_terrain == True
    assert a.output_plots == True
    assert a.output_terminal_summary == True
    assert a.accordance_measures == []
    assert a.exp_time_extent == ['2016/01/01', '2016/12/31']
    pass

