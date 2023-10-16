import numpy as np

ThetaRes = [0.2 for i in range(18)] + [0.002 for i in range(16)]
print('! 2m Peat on Rock')

ThetaSat = ', '.join(['0.85' for i in range(18)] + ['0.05' for i in range(16)])
print(f'ThetaSat = {ThetaSat}')

AlphaVanGenuchten = ', '.join(['0.03' for i in range(18)] + ['0.001' for i in range(16)])
print(f'AlphaVanGenuchten = {AlphaVanGenuchten}')

NVanGenuchten = ', '.join(['1.8' for i in range(18)] + ['1.2' for i in range(16)])
print(f'NVanGenuchten = {NVanGenuchten}')

NormalHydrConductivity = ', '.join(['0.3' for i in range(18)] + ['0.000001' for i in range(16)])
print(f'NormalHydrConductivity = {NormalHydrConductivity}')

LateralHydrConductivity = ', '.join(['0.3' for i in range(18)] + ['0.000001' for i in range(16)])
print(f'LateralHydrConductivity = {LateralHydrConductivity}')

TfreezingSoil = ', '.join(['-0.1' for i in range(18)] + ['0' for i in range(16)])
print(f'TfreezingSoil = {TfreezingSoil}')

ThermalConductivitySoilSolids = ', '.join(['2.5' for i in range(18)] + ['2.5' for i in range(16)])
print(f'ThermalConductivitySoilSolids = {ThermalConductivitySoilSolids}')

ThermalCapacitySoilSolids = ', '.join(['2250000' for i in range(18)] + ['2250000' for i in range(16)])
print(f'ThermalCapacitySoilSolids = {ThermalCapacitySoilSolids}')

