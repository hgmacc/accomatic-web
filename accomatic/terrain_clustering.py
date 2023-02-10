import os
import sys
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
from matplotlib import patches
from scipy.stats import chi2
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.simplefilter(action='ignore')

def getval(dat, lon, lat):
    """
    Gets value from tif data. Must first convert lat/lon coordinate to 3413 CRS. 

    Args:
        dat (rasterio data): array from tif file
        lon (float): longitude coordinate
        lat (float): latitude coordinate

    Returns:
        float: Value in dat at lat/lon coordinate.
    """
    z = dat.read()[0]
    to_utm = pyproj.Transformer.from_crs(4326, 3413, always_xy=True) 
    a, b = to_utm.transform(lon, lat)
    idx = dat.index(a, b)
    return z[idx]

def populate_df_col(place, abbrev, coords):
    """
    Iterates through .tif files for a given location and getval() for each site.

    Args:
        place (string): Yellowknife, Lac de Gras or KDI
        abbrev (string): The abbrev at the beginning of sitenames to subset coordinate df

    Returns:
        pd.DataFrame: a dataframe with twi, con and iso columns added
    """
    df = coords[coords['name'].str.contains(abbrev)]
    to_do = ['twi','con','iso']
    for thing in to_do:
        path = f"/home/hma000/storage/terrain_exp/dem_processed/{place}/{place}_dem_10m_{thing}.tif"
        dat = rio.open(path)
        df[thing] = df.apply(lambda x: getval(dat, float(x['lon']), float(x['lat'])), axis=1)
        dat.close()    
    return df

def build_terrain_csv():
    """
    For each location, build df than merge them all at the end and write data to a csv file. 
    """
    coords = pd.read_csv('/home/hma000/storage/terrain_exp/ykl_coords.csv', usecols = ['name','lat','lon','elevation_in_metres','sky_view'])

    places= ['yk', 'kdi', 'ldg']
    abbrevs = ['YK','KDI','NGO']
    
    l_df = []
    for place, abbrev in zip(places, abbrevs):
        l_df.append(populate_df_col(place, abbrev, coords))
    df = pd.concat(l_df)
    df.to_csv('/home/hma000/storage/terrain_exp/ykl_terrain.csv', columns=['name','elevation_in_metres','twi','con','iso'])


def normalize_data(data):
    for col in ['twi', 'con','iso']:
        data[col] = (data[col]-data[col].min())/(data[col].max()-data[col].min())
    return data
    
def calculateMahalanobis(y=None, data=None, cov=None):
    y_mu = y - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()

def mahal_cleaning(data, p_val):
    data['mahal'] = calculateMahalanobis(y=data.iloc[:,1:], data=data.iloc[:,1:])
    data['p'] = 1 - chi2.cdf(data['mahal'], 3)
    data = data[data.p > p_val]
    return(data)

def elbow_plotting(X):
    wcss = []
    for i in range(1,11):
        k_means = KMeans(n_clusters=i,init='k-means++', random_state=42)
        k_means.fit(X)
        wcss.append(k_means.inertia_)
    # Plot elbow curve
    plt.plot(np.arange(1,11),wcss)
    plt.xlabel('Clusters')
    plt.ylabel('SSE') # sum of squares error
    plt.savefig('/home/hma000/accomatic-web/recycling/terrain_clustering/k_means_elbow_curve_norm_mahal.png')


def knn_clustering(K_VALUE = 0):
    for p_val in [0.01]:
        data = pd.read_csv('/home/hma000/storage/terrain_exp/ykl_terrain.csv', usecols = ['name','twi','con','iso'])
        #data = normalize_data(data)
        data = mahal_cleaning(data, p_val)
        X = data.iloc[:,1:].values
        
        k_means_optimum = KMeans(n_clusters = K_VALUE, init = 'k-means++',  random_state=42)
        y = k_means_optimum.fit_predict(X)
        
        data['cluster'] = y  
        data1 = data[data.cluster==0]
        data2 = data[data.cluster==1]
        if K_VALUE > 2: data3 = data[data.cluster==2]
        if K_VALUE > 3: data4 = data[data.cluster==3]
        
        kplot = plt.axes(projection='3d')
        xline = np.linspace(0, 1, 1)
        yline = np.linspace(0, 1, 1)
        zline = np.linspace(0, 1, 1)
        kplot.plot3D(xline, yline, zline, 'black')
        kplot.scatter3D(data1['twi'], data1['con'], data1['iso'], c='red', label = 'Cluster 1')
        kplot.scatter3D(data2['twi'], data2['con'], data2['iso'],c ='green', label = 'Cluster 2')
        if K_VALUE > 2: kplot.scatter3D(data3['twi'], data3['con'], data3['iso'],c ='blue', label = 'Cluster 3')
        if K_VALUE > 3: kplot.scatter3D(data4['twi'], data4['con'], data4['iso'],c ='yellow', label = 'Cluster 4')

        plt.title(f'k means clustering (k = {K_VALUE}, n={data.shape[0]}, silouette score = {silhouette_score(X,y): .2f}, p_val = {p_val})')
        print(f'(k = {K_VALUE}, n={data.shape[0]}, silouette score = {silhouette_score(X,y)}, p_val = {p_val})')
        plt.savefig(f'/home/hma000/accomatic-web/recycling/terrain_clustering/k_{K_VALUE}_means_cluster_3D.png')
        data.to_csv('/home/hma000/storage/terrain_exp/ykl_clustered_terrains.csv', columns=['name','twi','con','iso', 'cluster'])
        
df = pd.read_csv("/home/hma000/storage/terrain_exp/ykl_clustered_terrains.csv", usecols=['name','twi','con','iso', 'cluster'])


# a = df[df['cluster'].astype('str').str.contains(i)]
print('< 0.33\n', df[df.con < 0.33].name)
print('\n0.33 - 0.66\n', df[df.con.between(0.335, 0.655)].name)
print('\n> 0.66\n', df[df.con > 0.66].name)

