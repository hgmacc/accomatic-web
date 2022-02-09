from accomatic.Settings import *


def test_settings():
    a = Settings('tests/test_data/test_toml_settings.toml')
    assert a.get_pickled_data() == True
    assert a.get_exp_season() == True
    assert a.get_exp_acco() == True
    assert a.get_exp_terrain() == True
    assert a.get_output_plots() == True
    assert a.get_output_terminal_summary() == True
    assert a.get_acco_list() == []
    assert a.get_exp_time_extent() == ['2016/01/01', '2016/12/31']
    pass

