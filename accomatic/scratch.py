import xarray as xr
import matplotlib.pyplot as plt

o = xr.open_dataset('/fs/yedoma/usr-storage/hma000/KDI/talikForcing/scaled_era5_1h_scf1.5.nc')
print(o.variables)
#o.[3, :].plot(color='#F50B00', linewidth=1.5)
plt.show()