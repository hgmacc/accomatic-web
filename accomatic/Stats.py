from operator import indexOf
import random
import seaborn as sns
import xarray as xr
from matplotlib.dates import DateFormatter
from accomatic.NcReader import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
from static.statistics_helper import *


def boot(stat, o, m, boot_size):
    nrows = range(o.shape[0])
    window = 5
    res = []
    
    for j in range(boot_size):
        ix = random.randint(nrows.start, nrows.stop-(window+1))

        a = acco_measures[stat](o.iloc[ix:ix+window],
                                m.iloc[ix:ix+window])
        res.append(a)
    return res


def run(o, m, exp, site, szn):
    d = {"data_avail": len(o)}
    for sim in m.columns:
        for stat in exp.acco_list:
            if len(o) < 30: d[stat] =  Data(np.full(exp.boot_size, np.nan))
            else: 
                if exp.boot_size: d[stat] = Data(boot(stat, o, m[sim], exp.boot_size))
                else: d[stat] = acco_measures[stat](o, m[sim])
        row = exp.res_index(site, sim, szn)
        exp.results.loc[row, list(d.keys())] = list(d.values())
        

def build(exp):
    for site in exp.sites_list:
        df = exp.obs(site).join(exp.mod(site)).dropna()
        o, m = df.obs.dropna(), df.drop(["obs"], axis=1)
        for szn in exp.szn_list:
            run(
                o[o.index.month.isin(time_code_months[szn])],
                m[m.index.month.isin(time_code_months[szn])],
                exp,
                site,
                szn,
            )

def rank(exp, csv_file_name='ranking.csv'):
    # get ranks from every subset: 
    df = exp.res()
    df['rank'] = np.nan
    df['rank_stat'] = np.nan    
    
    for stat in exp.acco_list:
        for terr in list(set(exp.terr_list)):
            for szn in exp.szn_list:
                tmp = df[df.szn==szn]
                tmp = tmp[tmp.terr==terr]

                rank_stat = pd.Series([float("{0:.3}".format(np.nanmean(i.v))) for i in tmp[stat]])
                rank = rank_stat.rank(method='max').tolist()
                rank_stat = rank_stat.tolist()
                
                for row, i in zip(tmp.index.tolist(), range(len(rank))):
                    df.loc[row, ['rank','rank_stat']] = [rank[i], rank_stat[i]]
                
    df[['sim','szn','terr','data_avail','rank','rank_stat']].to_csv(csv)


    
    