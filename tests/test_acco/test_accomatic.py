from accomatic.accomatic import *

def test_accomatic():
    mod_dir: str = 'tests/test_data/test_mod_dir'
    obs_csv: str = 'tests/test_data/test_mod_dir/test_mod_data.csv'
    set_toml: str = 'tests/test_data/test_toml_settings.toml'
    sit_csv: str = 'tests/test_data/test_sites_list.csv'

    assert accomatic(mod_dir, obs_csv, set_toml, sit_csv) == 1
    pass



