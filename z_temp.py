import netCDF4 as nc
import sys

a = '/home/hma000/storage/yk_kdi_ldg/era_pl/short_a.nc'
b = '/home/hma000/storage/yk_kdi_ldg/era_pl/short_b.nc'
c = '/home/hma000/storage/yk_kdi_ldg/era_pl/short_c.nc'
d = '/home/hma000/storage/yk_kdi_ldg/era_pl/short_d.nc'

l = ['era5_pl_ykl_sites_geotop_surface.nc',
'era5_pl_ykl_sites_geotop.nc',
'era5_sa_ykl_sites_geotop.nc',
'era5_sf_ykl_sites_geotop.nc',
'era5_to_ykl_sites_geotop.nc']

# ncks -d time,100000000000,100000000000 /home/hma000/storage/yk_kdi_ldg/interpolated/era5_pl_ykl_sites_geotop.nc /home/hma000/storage/yk_kdi_ldg/era_pl/short_d.nc

for file in l:
    nc_a = nc.Dataset(file)
    print(file, ': ', len(nc_a['time'])) #[0], nc_a['time'][-1])
    nc_a.close()


