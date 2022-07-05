from accomatic.Settings import *


def test_settings():
    a = Settings("tests/test_data/test_toml_settings.toml")
    assert type(a.model_nc_dir) == '/scratch/s/stgruber/hma000/gtpem_runs/kdi/KDI_JUNE20'
    assert type(a.exp_season) == bool
    assert type(a.exp_acco) == bool
    assert type(a.exp_terrain) == bool
    assert type(a.output_plots) == bool
    assert type(a.output_terminal_summary) == bool
    assert a.accordance_measures == []
    pass


