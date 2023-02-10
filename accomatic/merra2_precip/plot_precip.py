import pandas as pd

sites = pd.read_csv('/home/hma000/storage/yk_kdi_ldg/par/ykl.csv', usecols=['station_name','longitude_dd','latitude_dd'])
sites.longitude_dd = round((sites.longitude_dd * 2) / 2)
print(sites.longitude_dd.head())