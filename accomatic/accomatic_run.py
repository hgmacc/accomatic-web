from os import path

from NcReader import *
from Settings import Settings

exp = Settings("../tests/test_data/test_toml_settings.toml")

if exp.acco:
    m = Dataset(exp.model_pth)
    o = Dataset(exp.obs_pth)
    a = path.join(path.dirname(exp.model_pth), "acco.nc")
    print(a)
    create_acco_nc(a, exp)
