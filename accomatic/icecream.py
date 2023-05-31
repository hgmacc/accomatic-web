import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from Stats import willmott_refined_d

def willmott(mod, real):
    # From Luis (2020)
    o_mean = np.mean(real)
    a = sum(abs(mod - real))
    b = 2 * sum(abs(mod - o_mean))
    if a <= b:
        return 1 - (a / b)
    else:
        return (b / a) - 1

np.random.seed(1); n = 1000

pred = np.random.randint(low=-100, high=100, size=(n,10))
real = np.random.randint(low=-100, high=100, size=10) / 100

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

def sin(x, freq=120, amp=1, lag=0, off=0, noise=False):
    a = amp * np.sin((2.*np.pi/freq)*x+lag) + off 
    if noise: 
            noise_arr = np.random.randint(low=-100, high=100, size=len(x)) / 1000
            for i in np.random.randint(low=0, high=120, size=5) :
                noise_arr[i] = noise_arr[i] + (np.random.randint(low=5, high=15, size=1) / 10)
                print(noise_arr[i])
            b = noise_arr * np.sin((2.*np.pi/365*120)*x)
            return a + b
    else: return a
    
def test():
    # y = A*np.sin(omega*x+phase) + offset
    n = 3
    fig, ax = plt.subplots(n, sharey=True, figsize=(10, 5*n))
    x = np.array(range(120))
    
    models = {"$offset = 1$" : sin(x, off=1),
            "$offset = -1$" : sin(x, off=-1),
            "$A = 0.5$" : sin(x, amp=0.5),
            "$\Theta = 0.5$" : sin(x, lag=0.5),
            "$offset = 0.5$" : sin(x, off=0.5),
            "$Step-wise$" : np.around(sin(x) * 1.0) / 1.0,
            "$\Theta = 0.5$" : sin(x, lag=-0.5)}
    
    # REAL DATA
    real = sin(x, noise=True)
    ax[0].plot(x, real, 'k', linewidth=3, label="Real", zorder=5)
    ax[1].plot(x, real, 'k', linewidth=3, label="_Real_", zorder=5)
    ax[2].plot(x, real, 'k', linewidth=3, label="_Real_", zorder=5)
    
    # BIAS
    ax[0].plot(x, models["$offset = 1$"], c='#008080',label="$offset = 1$")
    ax[0].plot(x, models["$offset = -1$"], c="#f3700e", label="$offset = -1$")
    ax[0].legend()
    
    # AMPLITUDE
    ax[1].plot(x, models["$A = 0.5$"], c="#008080", label="$A = 0.5$")
    ax[1].plot(x, models["$\Theta = 0.5$"], c="#f3700e", label="$\Theta = 0.5$")
    ax[1].plot(x, models["$offset = 0.5$"], c="#f50b00", label="$offset = 0.5$")
    ax[1].legend()
    
    # ANGULAR VELOCITY
    ax[2].plot(x, models["$Step-wise$"], c="#008080",label="$Step-wise$")
    ax[2].plot(x, models["$\Theta = 0.5$"], c="#f3700e", label="$\Theta = 0.5$")
    ax[2].legend()
    
    # NO SNOW
    # ax[2].plot(x, np.concatenate((real[:60]+.05, real[60:]*1.5)))
    
    
    r = ["{:.2f}".format(np.corrcoef(mod, real)[0, 1]) for mod in models.values()]
    mae = ["{:.2f}".format(np.mean(np.abs(mod - real))) for mod in models.values()]
    rmse = ["{:.2f}".format(np.sqrt(np.mean((mod - real)**2))) for mod in models.values()]
    bias = ["{:.2f}".format(np.mean(mod - real)) for mod in models.values()]
    will = [ "{:.2f}".format(willmott_refined_d(mod, real)) for mod in models.values()]
    
    df = pd.DataFrame()

    for i in range(n):
        ax[i].set_yticks([i/100 for i in range(-100, 110, 50)], labels=[str(i/100) for i in range(-100, 110, 50)])
        ax[i].set_xticks([])
        
    ax[n-1].set_xticks([i*10 for i in range(13)], labels=[str(i) for i in range(13)])
    plt.legend()
    plt.savefig('/home/hma000/accomatic-web/tests/plots/30MAY/sin.png')

test()






"""
    # LOW AMPLITUDE
    m1 = 0.5 * np.sin((2.*np.pi/f)*x+0) + 0
    ax[0].plot(x, m1, c="#008080", label="Model 1 $r = 1.0$")
    
    # LAGGY 
    m2 = 1 * np.sin((2.*np.pi/f)*x-0.5) + 0
    ax[0].plot(x, m2, c="#f50b00", label="Model 2 $BIAS = 0.0$")
    
    # STEP WISE CURVE
    m3 = np.concatenate((np.full(shape=15, fill_value=0.5),
                         np.full(shape=30, fill_value=1),
                         np.full(shape=30, fill_value=0),
                         np.full(shape=30, fill_value=-1),
                         np.full(shape=15, fill_value=-0.5)))
    ax[0].plot(x, m3, c="#1ce1ce", label="Model 3 $RMSE = 0.26$")
    
    # NO SNOW
    m4 = np.concatenate((real[:60]+.05, real[60:]*1.5))
    ax[0].plot(x, m4, c="#f3700e", label="Model 4 $d_1 = 0.86")

    # ANGULAR VELOCITY
    m5 = 1 * np.sin((2.*np.pi/f)*x) + (0.15*np.sin((2.*np.pi/365*120)*x))
    ax[0].plot(x, m5, c="#f50b00", label="Model 5 $ = 0.0$")
    
    # SQUIGGLES
    m5 = 1 * np.sin((2.*np.pi/120)*x) + (0.1*np.sin((2.*np.pi/365*120)*x))
    ax[3].plot(x, m5, label="Squiggly")
    
    
    models = [m1, m2, m3, m4, m5]

    r = ["{:.2f}".format(np.corrcoef(mod, real)[0, 1]) for mod in models]
    mae = ["{:.2f}".format(np.mean(np.abs(mod - real))) for mod in models]
    rmse = ["{:.2f}".format(np.sqrt(np.mean((mod - real)**2))) for mod in models]
    bias = ["{:.2f}".format(np.mean(mod - real)) for mod in models]
    will = [ "{:.2f}".format(willmott_refined_d(mod, real)) for mod in models]
    
    for mod in models:
        print("\n\nr: ", "{:.2f}".format(np.corrcoef(mod, real)[0, 1]))
        print("mae: ", "{:.2f}".format(np.mean(np.abs(mod - real))))
        print("rmse: ", "{:.2f}".format(np.sqrt(np.mean((mod - real)**2))))
        print("bias: ", "{:.2f}".format(np.mean(mod - real)))
        print("willmott: ", "{:.2f}".format(willmott_refined_d(mod, real)))
"""