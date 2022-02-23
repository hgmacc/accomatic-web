from accomatic.accomatic import *


def test_accomatic():
    model_dir: str = 'tests/test_data/test_mod_dir'
    obs_pth: str = 'tests/test_data/test_obs_dir/test_obs.pickle'
    set_toml: str = 'tests/test_data/test_toml_settings.toml'
    sit_csv: str = 'tests/test_data/test_sites_list.csv'

    assert accomatic(model_dir, obs_pth, set_toml, sit_csv) == 1
    pass



