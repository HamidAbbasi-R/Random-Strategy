#%%
from modules import RandomStrategy
from modules import create_plots
from modules import obtain_OHLC
from modules import ADX
from modules import find_max_prob_profit as fmpp
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def single_simulation(
    asset,
    start_time,
    duration,
    interval,
    initial_capital = 50000,        # initial capital in USD
    N_test = 200,
    RRR = 2,            # reward to risk ratio
    risk_per_trade = 1,     # risk per trade in percentage
    P_win = 0.5,        # probability of win
    trailing_stop = True,
    cut_off = None,          # cut_off = [1.05,0.95]
    true_leverage = 2,
    ADX_threshold = 30,
    ADX_condition = 'trade-above',
    ## VISUALIZATION PARAMETERS
    debug           = False,
    show_figs       = False,
    oneD_hist       = False,
    twoD_hist       = False,
    durations_hist  = False,
    capital_curve   = False,
    save_results    = False,
    ):
        
    # end time is the start time plus the duration 
    # assume duration is in the format of '1D', or '4h, or '5m', etc
    end_time = pd.to_datetime(start_time) + pd.Timedelta(duration)

    # obtain the OHLC data
    OHLC = obtain_OHLC(asset, interval, start_time, end_time)
    adx_values = ADX(OHLC['Close'].values, OHLC['High'].values, OHLC['Low'].values, adx_period=14)
    indicators = {'ADX': adx_values}

    # simulate the strategy N_test times
    N_test = 1 if debug else N_test
    out = [None]*N_test
    for i in tqdm(range(N_test)):
        out[i] = RandomStrategy(
            OHLC=OHLC,
            asset=asset,
            risk_per_trade=risk_per_trade,
            reward_to_risk_ratio=RRR,
            True_leverage=true_leverage,
            P_win=P_win,
            ADX_threshold=ADX_threshold,
            ADX_condition=ADX_condition,
            cut_off=cut_off,
            trailing_stop=trailing_stop,
            initial_capital=initial_capital,     
            indicators=indicators,
            show_fig = debug,
            show_trades= debug,
        )
    
    # find the highest probability of profit
    PP, region = fmpp(out, initial_capital)

    if show_figs and not debug:
        create_plots(
            N_test, 
            out, 
            initial_capital = initial_capital,
            cut_off = cut_off,
            hist_durations = durations_hist,
            oneD_hist = oneD_hist,
            twoD_hist = twoD_hist,
            capital_curve = capital_curve,
            show_candles=False,
            # indicators=indicators
            )

    # save the results on disk
    if save_results and not debug:
        np.save('test_results.npy', out)

    # load the results
    # out = np.load('test_results.npy', allow_pickle=True)

    return PP, region