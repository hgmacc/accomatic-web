import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from static.statistics_helper import time_code_months
import sys 

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "10"



import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def spider_plot_normalize(df_col):
    return((df_col - df_col.min()) / (df_col.max() - df_col.min()))

def get_data(df, terrain_list, statistic_list, simulation_list):
    
    # Getting rid of low data and other depth results
    df = df[df.data_avail > 250]
    df = df[df.depth == 10].drop(columns=['depth']) 

    # Selecting key columns, interpolate rank_stat over months with missing data (undo this later)
    df = df[['sim','szn','terr','rank_stat','stat']]
    df['rank_stat'] = df.rank_stat.interpolate()
    spider_list = [['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']]

    # Normalizing stat values
    for statistic in statistic_list:
        # Select rows 
        rows = df[df.stat == statistic].index
        # Normalize statistical values
        nrml = spider_plot_normalize(df[df.stat == statistic].rank_stat).to_list()
        # Assign new values
        df.loc[rows, 'rank_stat'] = nrml

    # ACCORDANCE ROWS: 1(acco) x n(terrains)
    for statistic in statistic_list:
        # df_stat = (1) STAT, (n) terrain, (n) models, (12) months
        df_stat = df[df.stat == statistic].drop(columns=['stat'])
        for terrain in terrain_list: 
            # df_terr = (1) stat, (1) TERRAIN, (n) models, (12) months
            df_terr = df_stat[df_stat.terr == terrain].drop(columns=['terr'])
            
            data = [] # A list of lists; second part of spider_list tuple entry
            for simulation in simulation_list:
                # df_sim = (1) stat, (1) terrain, (1) MODEL, (12) months
                df_sim = df_terr[df_terr.sim == simulation].drop(columns=['sim'])
                # Organize (12) months in order
                df_sim['szn_no'] = [time_code_months[i][0] for i in df_sim.szn]
                df_sim = df_sim.sort_values('szn_no').drop(columns=['szn_no'])
                result = df_sim.rank_stat.to_list()
                if len(df_sim.rank_stat) < 12:
                    result = [0,0,0,0,0,0,0,0,0,0,0,0]
                data.append(result)
            
            spider_list.append((f'{statistic}+{terrain}', data))
        
        stat_mean_data = []
        for simulation in simulation_list:
            # df_sim = (1) stat, (n) terrain, (1) model, (12) months
            df_sim = df_stat[df_stat.sim == simulation].drop(columns=['sim'])
            df_sim = df_sim.groupby('szn').mean().drop(columns=['terr']).reset_index(drop=False)

            # Organize (12) months in order
            df_sim['szn_no'] = [time_code_months[i][0] for i in df_sim.szn]
            df_sim = df_sim.sort_values('szn_no').drop(columns=['szn_no'])
            result = df_sim.rank_stat.to_list()
            if len(df_sim.rank_stat) < 12:
                result = [0,0,0,0,0,0,0,0,0,0,0,0]
            stat_mean_data.append(result)

        spider_list.append((f'{statistic}+mean', stat_mean_data))
        
    # LAST ROW: TERRAIN MEANS: mean(n(acco)) x 1(terrains)
    for terrain in terrain_list: 
        # df_terr = (n) stats, (1) TERRAIN, (n) models, (12) months
        df_terr = df[df.terr == terrain].drop(columns=['terr'])
            
        data = [] # A list of lists; second part of spider_list tuple entry
        for simulation in simulation_list:
            # df_sim = (n) stats, (n) terrain, (1) MODEL, (12) months
            df_sim = df_terr[df_terr.sim == simulation].drop(columns=['sim'])
            df_sim = df_sim.groupby('szn').mean().reset_index(drop=False)
            df_sim['szn_no'] = [time_code_months[i][0] for i in df_sim.szn]
            df_sim = df_sim.sort_values('szn_no').drop(columns=['szn_no'])
            result = df_sim.rank_stat.to_list()
            if len(df_sim.rank_stat) < 12:
                result = [0,0,0,0,0,0,0,0,0,0,0,0]
            data.append(result)
        spider_list.append((f'{terrain}+mean', data))
    
    # LAST CELL: mean(n(acco)) x mean(n(terrains))
    data = []
    tmp = {}
    for simulation in simulation_list:
        # df_sim = (n) stats, (n) terrain, (1) MODEL, (12) months
        df_sim = df[df.sim == simulation].drop(columns=['sim', 'terr'])
        df_sim = df_sim.groupby('szn').mean().reset_index(drop=False)

        df_sim['szn_no'] = [time_code_months[i][0] for i in df_sim.szn]
        df_sim = df_sim.sort_values('szn_no').drop(columns=['szn_no'])
        result = df_sim.rank_stat.to_list()
        if len(df_sim.rank_stat) < 12:
            result = [0,0,0,0,0,0,0,0,0,0,0,0]
        data.append(result)
        try: 
            a = sum(result) / len(result)
        except :
            a = 0
        tmp[simulation] = a
        
    spider_list.append((f'{terrain}+mean', data))
    return spider_list
    
if __name__ == '__main__':
        
    df = pd.read_csv('/home/hma000/accomatic-web/tests/test_data/csvs/ranking/ranking_combined.csv')
    
    terrain_list = [1, 2, 8, 15] #df_og.terr.unique()
    statistic_list = ['R', 'WILL', 'MAE'] # df_og.stat.unique()
    simulation_list = ['ens', 'era5', 'jra55', 'merra2'] # df_og.sim.unique()
    
    data = get_data(df, terrain_list, statistic_list, simulation_list)

    N = 12
    theta = radar_factory(N, frame='polygon')

    spoke_labels = data.pop(0)
    cols, rows = (len(terrain_list)+1), (len(statistic_list)+1)
    fig, axs = plt.subplots(figsize=(cols*4, rows*4), 
                            ncols=cols,
                            nrows=rows,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.40, hspace=0.30, top=0.9, bottom=0.05)

    colors = ['#59473c', '#008080','#f50b00','#F3700E']

    rgrid_toggle = True
    # Plot the 15 cases from the example data on separate axes
    for ax, (title, case_data) in zip(axs.flat, data):
        ax.set_facecolor("white")

        ax.set_rgrids([])
        # So we don't label every single plot
        if rgrid_toggle: 
            ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
            rgrid_toggle = False
        
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels)

    fig.text(0.5, 0.965, 'Seasonal performance of simulations with normalized statical values.',
             horizontalalignment='center', color='black', weight='bold',
             size='xx-large')
    
    # Column headers
        
    statistic_list.append('Mean')
    for col, place in zip(statistic_list,  [(rows-i)/(rows+1) for i in range(rows)]):
        fig.text(0.05, place, col, color='black', weight='bold', size='medium')
    
    # Row headers
    terrain_list.append('Mean')
    for col, place in zip(terrain_list, [0.99-(cols-i)/(cols+1) for i in range(cols)]):
        fig.text(place, 0.925, f'Terrain {col}', color='black', weight='bold', size='medium')
    
    
    # (x,x) (y,y)
    line_vert = plt.Line2D((.76,.76),(.05,.9), color="k", linewidth=1)
    line_hor = plt.Line2D((.1,.925),(.25,.25), color="k", linewidth=1)
    fig.add_artist(line_vert)
    fig.add_artist(line_hor)
    
    
    plt.savefig('/home/hma000/accomatic-web/plots/spider/spider_big_proto.png')#, transparent=True)
    