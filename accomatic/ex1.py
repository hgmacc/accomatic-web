from Experiment import *
from Plotting import * 
import statistics as stats
import matplotlib.pyplot as plt
from static.statistics_helper import *
import time

def bootstrap(res):

    bs = []
    bs_t = []
    for i in range(10000):
        bootstrap_sample = random.choices(res, k=len(res))
        bs.append(bs.append(stats.mean(bootstrap_sample) - stats.mean(np.array(res))))
        bs_t.append(stats.mean(bootstrap_sample) - stats.mean(np.array(res)) / stats.sem(bootstrap_sample))
        
    bs = bs[500:9500]
    
    plt.hist(bs, histtype='step', color='red', label='regular bs')
    plt.hist(bs_t, histtype='step', color='blue', label='t statistic')
    plt.legend()
    plt.savefig('histogram.png')
    
    return True
    

if __name__ == "__main__":
    exp = Experiment('/home/hma000/accomatic-web/tests/test_data/toml/run.toml')
    o = exp.obs('ROCKT1')
    m = exp.mod('ROCKT1')
    stat = 'MAE'
    
    nrows = range(o.shape[0])
    res = []
    window = 10
    print('starting')
    start = time.time()

    while (len(res)<10) and ((time.time() - start) < 10):
        ix = random.randint(nrows.start, nrows.stop-(window+1))
        try:
            a = acco_measures[stat](o.iloc[ix:ix+window],
                                    m.iloc[ix:ix+window])
        except ValueError:
            continue
        
        res.append(a)
    print(res)
    bootstrap(res)
