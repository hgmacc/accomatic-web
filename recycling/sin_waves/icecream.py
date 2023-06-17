import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from Stats import willmott_refined_d

plt.rcParams["font.size"] = "16"

def willmott(mod, real):
    # From Luis (2020)
    o_mean = np.mean(real)
    a = sum(abs(mod - real))
    b = 2 * sum(abs(mod - o_mean))
    if a <= b:
        return 1 - (a / b)
    else:
        return (b / a) - 1

def nse_stat(prediction, observation):
    o_mean = observation.mean()
    a = sum(abs(observation - prediction) ** 2)
    b = sum(abs(observation - o_mean) ** 2)
    return 1 - (a / b)



np.random.seed(1); n = 1000

# pred = np.random.randint(low=-100, high=100, size=(n,10))
# real = np.random.randint(low=-100, high=100, size=10) / 100

x = range(10)
xnew = np.linspace(-1, 1, 300)  

class Data:
    _v: np.array
    
    def __init__(self, v):
        self._v = v / 100
        
    @property
    def v(self) -> np.array:
        return self._v
    
    @property
    def p(self) -> np.array:
        spl = make_interp_spline(range(10), self.v, k=3)
        return(spl(np.linspace(-1, 1, 300)))
    
    def __repr__(self):
        return repr(self.v)
    
def normal():
    r = [np.corrcoef(pred[i], real)[0, 1] for i in range(n)]
    mae = [np.mean(np.abs(pred[i] - real)) for i in range(n)]
    bias = [np.mean(pred[i] - real) for i in range(n)]
    pred_list = [Data(i) for i in pred]

    df = pd.DataFrame(columns = ['r','mae', 'bias', 'pred'])
    df = df.append(pd.DataFrame({'r': r, 'mae' : mae, 'bias' : bias, 'pred' : pred_list}),
                ignore_index = True)

    plt.plot(real, label='Observed', c='k')
    plt.plot(df.loc[df['r'].idxmax()].pred.v, label='Model 1', c='#008080',  linestyle=':')
    plt.plot(df.loc[df['mae'].idxmin()].pred.v, label='Model 2', c='#008080',  linestyle='-.')
    bias = [int(real.mean().round()) for i in range(10)]
    plt.plot(bias, label='Model 3', c='#008080',  linestyle='--')


    plt.legend()
    plt.savefig("normal.png")

def spline():
    r = [np.corrcoef(pred[i], real)[0, 1] for i in range(n)]
    mae = [np.mean(np.abs(pred[i] - real)) for i in range(n)]
    bias = [np.mean(pred[i] - real) for i in range(n)]
    pred_list = [Data(i) for i in pred]

    df = pd.DataFrame(columns = ['r','mae', 'bias', 'pred'])
    df = df.append(pd.DataFrame({'r': r, 'mae' : mae, 'bias' : bias, 'pred' : pred_list}),
                ignore_index = True)

    spl = make_interp_spline(x, real, k=3)  # type: BSpline
    plt.plot(spl(xnew), label='Observed', c='k')

    r_new = df.loc[df['r'].idxmax()].pred.p
    plt.plot(r_new, label='Model 1', c='#008080',  linestyle=':')
    # plt.scatter(x = range(10), y = df.loc[df['r'].idxmax()].pred.v)

    mae_new = df.loc[df['mae'].idxmin()].pred.p
    plt.plot(mae_new, label='Model 2', c='#008080',  linestyle='-.')

    bias_new = [int(real.mean().round()) for i in range(10)]
    spl = make_interp_spline(x, bias_new, k=3)  # type: BSpline
    plt.plot(spl(xnew), label='Model 3', c='#008080',  linestyle='--')

    plt.legend()
    plt.savefig("icecream.png")

plt.clf()

def build_df(x, datasets):
        
    r = ["{:.2f}".format(np.corrcoef(mod, sin(x))[0, 1]) for mod in datasets.values()]
    r2 = ["{:.2f}".format(float(i)**2) for i in r]
    mae = ["{:.2f}".format(np.mean(np.abs(mod - sin(x)))) for mod in datasets.values()]
    rmse = ["{:.2f}".format(np.sqrt(np.mean((mod - sin(x))**2))) for mod in datasets.values()]
    mae_outlier = ["{:.2f}".format(np.mean(np.abs(mod - datasets["real_outliers"]))) for mod in datasets.values()]
    nse = ["{:.2f}".format(nse_stat(mod, datasets["real_zero"])) for mod in datasets.values()]
    sum_error = ["{:.2f}".format(np.sum(np.abs((mod - sin(x))))) for mod in datasets.values()]
    sum_error_outlier = ["{:.2f}".format(np.sum(np.abs((mod - datasets["real_outliers"])))) for mod in datasets.values()]
    rmse_outlier = ["{:.2f}".format(np.sqrt(np.mean((mod - datasets["real_outliers"])**2))) for mod in datasets.values()]
    bias = ["{:.2f}".format(np.mean(mod - sin(x))) for mod in datasets.values()]
    will = [ "{:.2f}".format(willmott_refined_d(mod, sin(x))) for mod in datasets.values()]
    
    df = pd.DataFrame(columns = ['model', 'r', 'r2','mae', 'mae_outlier', 'rmse', 'sum_error_outlier',
                                'rmse_outlier', 'bias', 'will', 'sum_error', 'nse'])
    df = df.append(pd.DataFrame({'model' : datasets.keys(), 'r': r, 'r2':r2,
                                 'mae' : mae, 'rmse' : rmse, 'rmse_outlier' : rmse_outlier,
                                 'bias' : bias, 'will' : will, 'sum_error' : sum_error, 'sum_error_outlier' : sum_error_outlier, 'mae_outlier' : mae_outlier, "nse":nse}),
                                ignore_index = True)
    df = df.set_index(["model"]).T.astype(float)
    #print(df.to_latex())

    return df

def sin(x, freq=120, amp=10, lag=0, off=0, noise=False, noise_val = 0.5, outliers=False,  outliers_val = [10, 25]):
    
    a = amp * np.sin((2.*np.pi/freq)*x+lag) + off 
    if noise: 
            noise_arr = np.random.randint(low=noise_val*-10, high=noise_val*10, size=len(x)) / 10
            tmp = []
            a = a + noise_arr * np.sin((2.*np.pi/365*120)*x)
    if outliers and outliers_val == [10, 25]:
        for i in [10, 26, 30, 42, 50, 70, 72, 85, 90, 110]:
            a[i] = a[i] + (np.random.randint(low=-outliers_val[1], high=outliers_val[1], size=1))
    if outliers and outliers_val != [10, 25]: 
        for i in np.random.randint(low=0, high=outliers_val[0], size=120):
            a[i] = a[i] + (np.random.randint(low=-outliers_val[1], high=outliers_val[1], size=1))
    return a
    
def test():
    # y = A*np.sin(omega*x+phase) + offset
    
    n = 3
    fig, ax = plt.subplots(n, figsize=(8, 5*n)) 
    plt.tight_layout()
    x = np.array(range(120))
    
    datasets = {"pos_offset" : sin(x, off=5),
            "neg_offset" : sin(x, off=-5),
            "reduced_amp" : sin(x, amp=2.5),
            "lag_0" : sin(x, lag = 1.5, amp=1) * -0.01,
            "lag" : sin(x, lag = 0.25),
            "offset = 0.5" : sin(x, off=0.5),
            "step_wise" : np.around(sin(x) * 1/6) / (1/6),
            "low_corr" : sin(x, outliers=True,  outliers_val = [40, 10]),
            "real" : sin(x, noise=True),
            "real_outliers" : sin(x, noise=True, outliers=True),
            "real_zero": sin(x, lag = 1.5, amp=1) * 0.01 + np.random.randint(low=-10, high=10, size=120) / 10000}

    datasets['low_rmse'] = sin(x) + 2 # np.copy(datasets["real_outliers"]) + 2
    datasets["sin_wave"] = sin(x) # np.copy(datasets["real_outliers"])
    datasets["no_snow"] = np.concatenate((sin(x)[:60]+.05, sin(x)[60:]*2))
    datasets["high_var"] = sin(x, noise=True, noise_val=5)
    
    for i in [10, 26, 30, 42, 50, 70, 72, 85, 90, 110]:
        datasets['low_rmse'][i] = datasets["real_outliers"][i]
        datasets["sin_wave"][i] = datasets["sin_wave"][i-1]
        cir = pat.Ellipse((i, datasets["real_outliers"][i]), width=2.5, height=1.6, alpha=0.5, color='#f50b00',fill=True)
        ax[0].add_patch(cir)
        
    # AMPLITUDE
    ax[0].plot(x, datasets['real_outliers'], 'k', linewidth=3, label="Observations")
    ax[0].plot(x, datasets["sin_wave"], c="#f3700e", linewidth=1.5, label="Model 1")
    ax[0].plot(x, datasets['low_rmse'], c='#008080', linewidth=1.5, label="Model 2")

    # BIAS
    ax[1].plot(x, datasets['real'], 'k', linewidth=3, zorder=5)
    ax[1].plot(x, datasets["pos_offset"], c='#f3700e')
    ax[1].plot(x, datasets["neg_offset"], c="#008080")
    ax[1].plot(x, datasets["step_wise"], c="#f50b00")
    
    # CORRELATION
    ax[2].plot(x, datasets['real'], 'k', linewidth=3, label="_Real_", zorder=4)
    ax[2].plot(x, datasets["lag"], c="#008080", label="lag", zorder=5)
    ax[2].plot(x, datasets["reduced_amp"], c="#f3700e")
    #ax[2].plot(x, datasets["low_corr"], c="#f50b00")

    df = build_df(x, datasets)
    
    ax0_df = df.loc[['mae_outlier','rmse_outlier','sum_error_outlier']][["sin_wave", "low_rmse"]]
    ax0_table = ax[0].table(cellText=(ax0_df.to_numpy()),
                     colWidths=[0.1] * 3,
                     rowLabels=['MAE','RMSE','Toal Error'],
                     colLabels=['Model 1','Model 2'],
                     colColours =["#f3700e","#008080"],
                     loc='upper right')
    ax0_table.set_fontsize(16)
    ax0_table.scale(1,1.5)
    
    ax1_df = df.loc[['mae','r','bias']][["pos_offset", "neg_offset", "step_wise"]]
    ax1_table = ax[1].table(cellText=ax1_df.to_numpy(),#np.around(ax1_df.to_numpy() * 1),
                     colWidths=[0.1] * 3,
                     rowLabels=['MAE','R','BIAS'],
                     colLabels=['Model 1','Model 2', 'Model 3'],
                     colColours =["#f3700e","#008080", "#f50b00"],
                     loc='upper right')
    ax1_table.set_fontsize(14)
    ax1_table.scale(1,1.5)
    
    ax2_df = df.loc[['r','r2', 'will', 'sum_error']][["reduced_amp", "lag"]]
    ax2_table = ax[2].table(cellText=ax2_df.to_numpy(), 
                     colWidths=[0.1] * 2,
                     rowLabels=['r     ','$r^2$', '$d$', 'Total error'],
                     colLabels=['Model 1','Model 2'],
                     colColours =["#f3700e","#008080"],
                     loc='upper right')
    ax2_table.set_fontsize(14)
    ax2_table.scale(1,1.5)
    
    alpha = ['A','B','C','D','E']
    for i, l in zip(range(n), alpha[:n]):
        if i > 0: ax[i].text(1, 15, l, fontsize=20, fontweight='bold')
        else: ax[i].text(1, 22, l, fontsize=20, fontweight='bold')
        ax[i].set_yticks([i for i in range(-20, 30, 10)])
        ax[i].set_xticks([])
    ax[n-1].set_xticks([i*10 for i in range(13)], labels=[str(i) for i in range(13)])
    plt.savefig('/home/hma000/accomatic-web/tests/plots/30MAY/sin.png')

test()
