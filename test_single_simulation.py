#%%
from main import single_simulation

single_simulation(
    asset = 'BNB-USD',
    start_time = '2024-12-09',
    duration = '8h',
    interval = '5m',
    initial_capital = 50000,        # initial capital in USD
    N_test = 50,
    RRR = 1,
    risk_per_trade = 0.5,
    P_win = 0.5,
    trailing_stop = False,
    cut_off = None,          # cut_off = [1.05,0.95]
    true_leverage = 2.5,
    ADX_threshold = 0,
    ADX_condition = 'trade-above',
    ## VISUALIZATION PARAMETERS
    debug           = False,
    show_figs       = True,
    oneD_hist       = False,
    twoD_hist       = False,
    durations_hist  = True,
    capital_curve   = True,
    save_results    = False,
    )

