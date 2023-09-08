from Experiment import *
from Plotting import * 
import statistics as stats
import matplotlib.pyplot as plt
from static.statistics_helper import *
import time
import scipy

palette_list = ['#3DF735', '#f50b00', '#8C0B90', '#C0E4FF', '#27B502', '#7C60A8', '#CF95D7', '#145JKH']

def bs_increase_ss(res_list, lens):
    bs = []
    bs_t = []
    for res in res_list:
        bs_res = []
        bs_res_t = []
        for i in range(10000):
            bootstrap_sample = random.choices(res, k=len(res))
            bs_val = (stats.mean(bootstrap_sample) - stats.mean(np.array(res)))
            bs_t_val = (stats.mean(bootstrap_sample) - stats.mean(np.array(res))) / scipy.stats.sem(bootstrap_sample)
            
            bs_res.append(bs_val)
            bs_res_t.append(bs_t_val)
        bs.append(bs_res)
        bs_t.append(bs_res_t)

    cols = palette_list[:len(bs)]
    #cols = ['#%06X' % random.randint(0, 0xFFFFFF) for i in range(len(bs))]
    for i in range(len(bs)):
        plt.hist(bs[i][500:9500], histtype='step', color=cols[i], label=lens[i])
        plt.hist(bs_t[i][500:9500], histtype='step', linestyle='dashed', color=cols[i], label=f'{lens[i]}_t')

    plt.legend()
    plt.xlim((-4, 4))
    plt.savefig('t_histogram.png')

def get_10day_res(exp, sample_size):
    o = exp.obs('ROCKT1')
    m = exp.mod('ROCKT1')['ens']
    stat = 'MAE'
    
    nrows = range(o.shape[0])
    res = []
    window = 10

    while (len(res) < sample_size):# and ((time.time() - start) < 10):
        ix = random.randint(nrows.start, nrows.stop-(window+1))

        try:
            a = acco_measures[stat](o.iloc[ix:ix+window],
                                    m.iloc[ix:ix+window])
        except ValueError:
            continue
        res.append(a)
    return res
    
def get_allsites_res(exp):
    stat = 'MAE'
    res = []
    print('starting')

    for site in exp.obs().index.get_level_values(1).unique():
        merge = exp.obs(site).join(exp.mod(site)['ens'])
        try:
            a = acco_measures[stat](merge['obs'], merge['ens'])
        except ValueError:
            continue
        res.append(a)
    return res

if __name__ == "__main__":
    exp = Experiment('/Users/hannahmacdonell/Documents/projects/accomatic-web/tests/test_data/toml/local/run.toml')

    res_list = []
    lens = [100, 1000]
    for i in lens:
        res_list.append(get_10day_res(exp, i))
    bs_increase_ss(res_list, lens)

