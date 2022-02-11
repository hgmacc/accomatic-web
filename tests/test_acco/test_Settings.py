from accomatic.Settings import *


def test_settings():
    a = Settings("tests/test_data/test_toml_settings.toml")
    assert type(a.pickled_data) == bool
    assert type(a.exp_season) == bool
    assert type(a.exp_acco) == bool
    assert type(a.exp_terrain) == bool
    assert type(a.output_plots) == bool
    assert type(a.output_terminal_summary) == bool
    assert a.accordance_measures == []
    pass
