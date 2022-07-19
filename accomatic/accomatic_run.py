from NcReader import *
from Settings import Settings

exp = Settings("../tests/test_data/test_toml_settings.toml")
nope = False

if exp.acco and nope:
    m = Dataset(exp.model_pth)
    o = Dataset(exp.obs_pth)
    a = path.join(path.dirname(exp.model_pth), "acco.nc")

    if not path.exists(a):
        create_acco_nc(a, exp)
        print(f'New accomatic nc file created at {a}')

    simulations = read_manifest(exp.manifest)

    if exp.acco:

        print(simulations[71:74])

        print(len(simulations[72:].site.unique()))
        print(len(simulations[72:].model.unique()))

        print(len(simulations[:72].site.unique()))
        print(len(simulations[:72].model.unique()))

        #run_acco(m, o, simulations)

    #print(m['geotop']['simulation'][:3])
    #print(simulations.head())

    # 6 simulations per site ( 3 param x 2 reanalysis data types)
    # monthly average of just jra data


o = xr.open_dataset('/home/hma000/obs.nc')
print(o['platform_id'][:])

