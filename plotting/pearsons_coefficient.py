import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from calendar import month_abbr

import sys
sys.path.append("/home/hma000/accomatic-web/accomatic/")
from Experiment import *
from NcReader import read_exp
# dat = exp.obs.loc[idx[['res'], :, :, 'R']].droplevel("mode")
# Plot all twelve months of obs vs ens for one site, one year & Calculate R
# Plot twelve seperate months in different colours, each with thier own R value. 

test_site = 'YK16-RH01'

exp = read_exp('/home/hma000/accomatic-web/data/pickles/09May_0.1_0.pickle')

o = exp.obs(sitename=test_site).reset_index()
m = exp.mod(sitename=test_site).reset_index()

merged_df = pd.merge(o, m, left_on='index', right_on='time').drop(columns='time')
merged_df.set_index(pd.to_datetime(merged_df['index']), inplace=True)
merged_df.drop(columns='index', inplace=True)
merged_df = merged_df[merged_df.index.year == 2020][['obs','ens']]

# Function to plot linear regression line and R value
def plot_regression(ax, x, y, color, month=99):
    slope, intercept, r_value, _, _ = linregress(x, y)
    if month == 99: 
        month_abb = ''
    else: 
        month_abb = month_abbr[month]
    
    slope, intercept, r_value, _, _ = linregress(x, y)
    ax.plot(x, intercept + slope * x, label=f'{month_abb} (R² = {r_value**2:.2f})', color=color)
    

# Setup two subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: All data points with global linear regression line
sns.scatterplot(ax=axes[0], x=merged_df['ens'], y=merged_df['obs'], color='blue')
plot_regression(axes[0], merged_df['ens'], merged_df['obs'], color='blue')
axes[0].set_title('All Data Points with Global Linear Regression')
axes[1].legend(loc='best', frameon=False)

# Plot 2: Color-coded by month with individual linear regressions
for month in range(1, 13):
    month_data = merged_df[merged_df.index.month == month]
    color = sns.color_palette('dark', 12)[month - 1]
    sns.scatterplot(ax=axes[1], x=month_data['ens'], y=month_data['obs'], color=color)
    plot_regression(axes[1], month_data['ens'], month_data['obs'], color=color, month=month)

axes[1].set_title('Data Points Color-coded by Month with Monthly Linear Regressions')

# Set same xlim and ylim for both plots
common_lim = (merged_df[['ens', 'obs']].min().min(), merged_df[['ens', 'obs']].max().max())
axes[0].set_xlim(common_lim); axes[0].set_ylim(common_lim)
axes[1].set_xlim(common_lim); axes[1].set_ylim(common_lim)

# Clean up the legend in plot 2
handles, labels = axes[1].get_legend_handles_labels()
unique_labels = sorted(set(labels), key=lambda x: labels.index(x))
unique_handles = [handles[labels.index(label)] for label in unique_labels]
axes[1].legend(handles=unique_handles, loc='best', frameon=False)


# Ensure the left plot retains its legend
axes[0].legend(loc='best', frameon=False)

# Show the plots
plt.tight_layout()
plt.savefig(f'plotting/out/examples/pearsons{test_site}.png')
