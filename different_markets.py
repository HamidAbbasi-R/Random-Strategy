#%%
from main import single_simulation
import pandas as pd
import numpy as np
from tqdm import tqdm

# AVAILABLE ASSETS
# 'USDJPY=X' , 'USDCHF=X' , 'USDCAD=X', 
# 'EURUSD=X' , 'GBPUSD=X' , 'AUDUSD=X' , 'NZDUSD=X', 
# 'BTC-USD', 'ETH-USD', 'BNB-USD', 
# 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD'

asset = 'BNB-USD'
N_test = 200

# RISK MANAGEMENT PARAMETERS
initial_capital = 50000
risk_per_trade = 0.5
trailing_stop = True
RRR = 2
P_win = 0.9
true_leverage = 2.5

# TIME PARAMETERS
start_time = '2024-10-15 00:00:00'
end_time = 'now'
candles_length = '5m'              # timeframe of the candles in the OHLC data
# PP_chart_update_freq = '1d'         # frequency of the chart update
backward_simulation_duration = '4h'     # duration of the backward simulation

# ADX PARAMETERS
ADX_threshold = 0
ADX_condition = 'trade-above'

# DEBUG OPTIONS
debug = False


end_time = pd.to_datetime('today').strftime('%Y-%m-%d %H:%M:%S') if end_time == 'now' else end_time
# time_range = pd.date_range(start_time, end_time, freq=PP_chart_update_freq)
time_range = pd.date_range(start_time, end_time, freq=backward_simulation_duration)
out = [None] * len(time_range)
i=0
for time in tqdm(time_range):
    try:
        start_time = time - pd.Timedelta(backward_simulation_duration)
        out[i] = single_simulation(
            asset = asset,
            start_time = start_time,
            duration = backward_simulation_duration,
            interval = candles_length,
            initial_capital = initial_capital,
            N_test = N_test,
            RRR = RRR,
            risk_per_trade = risk_per_trade,
            P_win = P_win,
            trailing_stop = trailing_stop,
            true_leverage = true_leverage,
            ADX_threshold = ADX_threshold,
            ADX_condition = ADX_condition,
            debug = debug,
            show_figs=False,
            oneD_hist=False,
            twoD_hist=False,
            durations_hist=False,
            capital_curve=False,
            save_results=False,
            )
        if debug:
            break
    except:
        out[i] = [[np.nan,np.nan],[np.nan,np.nan]]
    i+=1


out = {
    'time_range': time_range, 
    'results': out, 
    'backward_simulation_duration': backward_simulation_duration,
}

np.save(f'./results/{asset}_results.npy', out, allow_pickle=True)
print('Results saved successfully!')