import numpy as np
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import warnings
import MetaTrader5 as mt5
from datetime import datetime
warnings.filterwarnings("ignore")
# mt5.initialize()

def obtain_OHLC(
    asset,           # asset to trade
    interval,                   # interval of price data [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
    start_time,                 # start time in the format 'YYYY-MM-DD'
    end_time,                   # end time in the format 'YYYY-MM-DD'
    random_market=False,        # use random price data
    ):
    
    ## price data
    if random_market:       # needs more work
        pass
        # initial_price = 1
        # price = np.zeros(max_steps)
        # price[0] = initial_price
        # price[1:] = price[0]+np.cumsum(
        #     np.random.uniform(
        #         price[0]*largest_price_change/100, 
        #         -price[0]*largest_price_change/100, 
        #         size=max_steps-1))
        # price = random_market_generator()
        # OHLC = pd.DataFrame({
        #     'Close': price,
        #     })

    else:
        OHLC = yf.Ticker(asset).history(
            # [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
            interval=interval,
            # period=Duration,
            start = start_time,
            end = end_time,
        )

        # only working with close data
        price = OHLC['Close'].values

        if len(price) == 0:
            raise ValueError('Not enough data points for trading!')

    return OHLC

def RandomStrategy(
        OHLC,                       # OHLC data
        asset,                      # asset to trade
        risk_per_trade=1,           # [%]
        reward_to_risk_ratio=5,     # [times]
        initial_capital=50000,      # initial capital [$]
        True_leverage=10,           # True leverage [times]
        Maximum_leverage=30,        # Maximum leverage [times]
        P_win=0.5,                  # probability of using the winning trade direction for the next trade
        ADX_threshold=25,           # ADX threshold for trading
        ADX_condition='trade-above',# ADX condition for trading
        trailing_stop = True,       # trailing stop loss
        cut_off=None,               # cut off values for ending the sessions
        seed=None,                  # seed for random price data
        indicators=None,            # indicators to use
        random_market=False,        # use random price data
        max_steps=100,              # number of steps for random price data
        largest_price_change=0.5,   # largest price change in percentage for random price data
        show_fig=False,
        show_trades=False,
    ):   
    
    theme = 'seaborn'

    # Working with OHLC data
    price = OHLC['Close'].values
    if not random_market:
        open = OHLC['Open'].values
        high = OHLC['High'].values
        low = OHLC['Low'].values
        close = OHLC['Close'].values
    
    max_steps = len(price)
    
    step=0
    SL_points = np.zeros(max_steps)
    TP_points = np.zeros(max_steps)
    trades_times = []
    position_size = []
    RT = risk_per_trade / 100
    RR = reward_to_risk_ratio
    inTrade = False
    capital = [initial_capital]
    trades = []
    trade_duration = []
    losses = 0
    wins = 0

    base_USD = ['USDJPY=X' , 'USDCHF=X' , 'USDCAD=X']

    quote_USD = ['EURUSD=X' , 'GBPUSD=X' , 'AUDUSD=X' , 
                 'NZDUSD=X', 'BTC-USD', 'ETH-USD', 
                 'BNB-USD', 'XRP-USD', 'ADA-USD', 
                 'SOL-USD', 'DOT-USD']

    # ADX indicator
    ADX = indicators['ADX'] if indicators is not None else None
    
    first_trade = True
    # how to set the seed for np.random
    np.random.seed(seed) if seed is not None else None
    while step<max_steps:
        if not inTrade:
            # do not trade if ADX cond is not met  
            trade_cond = ADX[step] > ADX_threshold if ADX_condition == 'trade-above' else ADX[step] < ADX_threshold
            if not trade_cond:   
                capital.append(capital[-1])
                SL_points[step] = np.nan
                TP_points[step] = np.nan
            else:
                # choose between buy and sell randomly
                # idea: make a better guess based on the last trade outcome
                # if first trade
                if first_trade:
                    # 1 for buy, -1 for sell
                    trades.append(np.random.choice([1, -1]))
                    first_trade = False
                else:

                    # assessing win or lose by comparing capital [not from hitting SL]
                    # when trailing SL is on, sometimes SL is hit but the trade is still profitable
                    # so it counts as a win, although SL is hit
                    previous_win_flag = capital[-1] > capital[-2]
                    p = [P_win, 1 - P_win] if previous_win_flag == (trades[-2] == 1) else [1 - P_win, P_win]
                    trades.append(np.random.choice([1, -1], p=p))

                trades_times.append(step)
                
                # position size should be in units of base currency
                if asset in quote_USD:
                    position_size.append(capital[-1] * True_leverage / price[step])
                    max_allowed_loss = RT * capital[-1]     # maximum allowed loss in capital units ($) - quote currency
                elif asset in base_USD:
                    position_size.append(capital[-1] * True_leverage)
                    max_allowed_loss = RT * capital[-1] * price[step]     # maximum allowed loss in quote currency
                else:
                    raise ValueError('asset not supported (only USD pairs are supported)')    
                
                SL = price[step] - max_allowed_loss/position_size[-1] if trades[-1]==1 else price[step] + max_allowed_loss/position_size[-1]
                TP = price[step] + RR * max_allowed_loss/position_size[-1] if trades[-1]==1 else price[step] - RR * max_allowed_loss/position_size[-1] 

                # SL and TP points for the plot
                SL_points[step] = SL
                TP_points[step] = TP

                inTrade = True
                capital.append(capital[-1])
                max_price_per_trade = high[step]
                min_price_per_trade = low[step]
            
        if inTrade:       # in trade

            if trailing_stop:   # adjusting trailing SL
                if trades[-1]==1:       # long 
                    if high[step] > max_price_per_trade:
                        max_price_per_trade = high[step]
                        SL = high[step] - max_allowed_loss/position_size[-1]
                elif trades[-1]==-1:    # short
                    if low[step] < min_price_per_trade:
                        min_price_per_trade = low[step]
                        SL = low[step] + max_allowed_loss/position_size[-1]
            
            SL_points[step] = SL
            TP_points[step] = TP

            # long hitting hard/trailing SL
            if trades[-1]==1 and low[step]<=SL:
                trades.append(-1)   # close the trade 
                trades_times.append(step)
                loss = position_size[-1] * (SL - price[trades_times[-2]])      # quote currency
                if asset in ['USDJPY=X' , 'USDCHF=X' , 'USDCAD=X']:
                    loss /= price[step]     # convert to base currency
                # capital.append(capital[-1] + loss)
                capital[-1] = capital[-1] + loss
                inTrade = False
                losses += 1
                trade_duration.append(step-trades_times[-2])
            
            # short hitting hard/trailing SL
            elif trades[-1]==-1 and high[step]>=SL:
                trades.append(1)    # close the trade
                trades_times.append(step)
                loss = position_size[-1] * (price[trades_times[-2]] - SL)      # quote currency
                if asset in ['USDJPY=X' , 'USDCHF=X' , 'USDCAD=X']:
                    loss /= price[step]     # convert to base currency
                # capital.append(capital[-1] + loss)
                capital[-1] =capital[-1] + loss
                inTrade = False
                losses += 1
                trade_duration.append(step-trades_times[-2])

            # long hitting TP
            elif trades[-1]==1 and high[step]>=TP:
                trades.append(-1)   # close the trade
                trades_times.append(step)
                gain = position_size[-1] * (price[trades_times[-1]] - price[trades_times[-2]])      # quote currency
                if asset in ['USDJPY=X' , 'USDCHF=X' , 'USDCAD=X']:
                    gain /= price[step]    # convert to base currency
                # capital.append(capital[-1] + gain)
                capital[-1] = capital[-1] + gain
                inTrade = False
                wins += 1
                trade_duration.append(step-trades_times[-2])

            # short hitting TP
            elif trades[-1]==-1 and low[step]<=TP:
                trades.append(1)    # close the trade
                trades_times.append(step)
                gain = position_size[-1] * (price[trades_times[-2]] - price[trades_times[-1]])      # quote currency
                if asset in ['USDJPY=X' , 'USDCHF=X' , 'USDCAD=X']:
                    gain /= price[step]    # convert to base currency
                # capital.append(capital[-1] + gain)
                capital[-1] = capital[-1] + gain
                inTrade = False
                wins += 1
                trade_duration.append(step-trades_times[-2])
            
            # in trade but not hitting SL or TP
            else:
                capital.append(capital[-1])

        if cut_off is not None and (capital[-1] < initial_capital * cut_off[1] or capital[-1] > initial_capital * cut_off[0]):
            break

        step += 1

    # adjust trade marker colors according to trade direction
    if show_fig:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        data = go.Candlestick(
            x       = np.arange(len(OHLC['Open'])),
            # x       = OHLC['time'],
            open    = OHLC['Open'],
            high    = OHLC['High'],
            low     = OHLC['Low'],
            close   = OHLC['Close'],
        )

        fig.add_trace(data)

        # show trades
        if show_trades:
            fig.add_trace(
                go.Scatter(
                    x=trades_times, 
                    y=price[trades_times], 
                    mode='markers', 
                    name='Trades',
                    # marker color red for sell and green for buy
                    marker=dict(color=['red' if trade==-1 else 'green' for trade in trades]),
                    ))
            
        # add capital plot
        fig.add_trace(
            go.Scatter(
                # x=np.arange(max_steps), 
                x=np.arange(len(capital)), 
                y=capital, 
                mode='lines', 
                name='Capital',
                line=dict(color='black'),
                ),
                secondary_y=True)
        
        # add SL and TP points
        fig.add_trace(
            go.Scatter(
                x=np.arange(max_steps), 
                y=SL_points, 
                mode='markers', 
                name='SL',
                marker=dict(color='red'),
                # marker size
                marker_size=2,
                ))
        fig.add_trace(
            go.Scatter(
                x=np.arange(max_steps), 
                y=TP_points, 
                mode='markers', 
                name='TP',
                marker=dict(color='green'),
                # marker size
                marker_size=2,
                ))

        # legend above plot
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                ),
            xaxis_rangeslider_visible=False,
            )
        
        fig.update_xaxes(title_text='Time')
        fig.update_yaxes(title_text='Price', secondary_y=False)
        fig.update_yaxes(title_text='Capital', secondary_y=True)
        # secondary y axis is log scale
        fig.update_layout(template=theme, yaxis2_type="log")
        fig.show()

    N_trades = len(trades_times)//2
    win_rate = wins/N_trades

    out = {
        'OHLC': OHLC,
        'N_trades': N_trades,
        'win_rate': win_rate,
        'capital': capital,
        'trade_duration': trade_duration,
        'max_steps': max_steps,
        # 'trades_times': trades_times,
        # 'capital_test': price[trades_times],
    }
    return out
    # print(f'Number of trades: {N_trades}')
    # print(f'Win rate: {win_rate*100:.2f} %')
    # print(f'Final capital: {final_capital:.2f}')
    # print(f'Average trade duration: {average_trade_duration:.2f} steps')

def create_plots(
        N_test,
        out,
        initial_capital = 50000,
        cut_off = None,
        capital_curve = True,
        show_candles = False,
        hist_durations = True,
        oneD_hist = False,
        twoD_hist = True,
        theme = 'seaborn',
        renderer = 'vscode',
        indicators = None,
        add_individual_trades_on_2Dhist = False,
        ):
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.io as pio
    pio.renderers.default = renderer

    # add the final capital of each test
    final_capitals = [out[i]['capital'][-1] for i in range(N_test)]
    final_times = [len(out[i]['capital']) for i in range(N_test)]
    # final_times = [out[i]['trades_times'][-1] for i in range(N_test)]
    final_N_trades = [out[i]['N_trades'] for i in range(N_test)]

    if hist_durations:
        trade_durations = [out[i]['trade_duration'] for i in range(N_test)]

        for i in range(N_test):
            # categorize trade durations into bins of 1 bar
            trade_durations[i] = [trade_durations[i].count(j) for j in range(max(trade_durations[i])+1)]
            # replace zeros with NaNs
            trade_durations[i] = [np.nan if trade_durations[i][j]==0 else trade_durations[i][j] for j in range(len(trade_durations[i]))]
        
        # create a heatmap of trade durations
        fig_trade_duration = go.Figure()
        fig_trade_duration.add_trace(
            go.Heatmap(
                z=trade_durations,
                x = np.arange(np.nanmax(trade_durations[0])+1),
                y = np.arange(N_test),
                colorscale='Viridis',
                colorbar=dict(title='Count'),
                ))
        
        # find the average trade duration for each test
        average_trade_duration = [np.average(out[i]['trade_duration']) for i in range(N_test)]
        fig_trade_duration.add_trace(
            go.Scatter(
                x=average_trade_duration,
                y=np.arange(N_test),
                mode='markers',
                marker=dict(color='red', size=3),
                showlegend=True,
                name='Average trade duration',
                ))

        fig_trade_duration.update_layout(
            title='Trade durations',
            xaxis_title='Trade duration [bars]',
            yaxis_title='Test number',
            template=theme,
            legend=dict(
                orientation="h",
                y = 1.1,
                x = 0,
                ),
            )
        fig_trade_duration.show()

    if capital_curve:
        fig_capital = make_subplots(
            rows=1, cols=2, 
            shared_yaxes=True, 
            horizontal_spacing=0.02,
            column_widths=[0.8, 0.2]
            )

        # capital curves
        for i in range(N_test):
            fig_capital.add_trace(
                go.Scatter(
                    x=np.arange(len(out[i]['capital'])), 
                    # x=out[i]['trades_times'],
                    # x = np.arange(out[i]['max_steps']),
                    y=out[i]['capital'],
                    mode='lines',
                    line=dict(color='rgba(31, 75, 156,0.1)'),
                    ),
                row=1, col=1
                )
        
        # add the initial capital line
        fig_capital.add_hline(
            y=initial_capital,
            line=dict(color='black', width=5),
            row=1, col=1,
            )

        fig_capital.add_trace(go.Histogram(
            y=final_capitals,
            # histnorm='count',
            orientation='h',
            marker=dict(color='rgba(31, 75, 156,1)'),
            showlegend=False,
            ), row=1, col=2)
        
        # draw the initial capital line in histogram as well
        fig_capital.add_hline(
            y=initial_capital,
            line=dict(color='black', width=5),
            row=1, col=2,
            )

        if show_candles:    
            data = go.Candlestick(
                # x       = out[0]['OHLC'].index,
                x       = np.arange(len(out[0]['OHLC']['Open'])),
                open    = out[0]['OHLC']['Open'],
                high    = out[0]['OHLC']['High'],
                low     = out[0]['OHLC']['Low'],
                close   = out[0]['OHLC']['Close'],
            )

            fig_capital.add_trace(
                data,
                row=1, col=1,
                secondary_y=True,)


        # count the number of winning scenarios
        if cut_off is not None:
            wins = sum([final_capitals[i] > initial_capital*cut_off[0] for i in range(N_test)])
            losses = sum([final_capitals[i] < initial_capital*cut_off[1] for i in range(N_test)])

        fig_capital.add_trace(
            go.Scatter(
                x=out[i]['max_steps']+1 * np.ones(N_test),
                y=final_capitals, 
                mode='markers',
                marker=dict(color='rgba(31, 75, 156,0.1)', size=5),
                ),
            )

        fig_capital.update_layout(
            title=f'Capital curves, N={N_test}',
            xaxis_title='Time',
            yaxis_title='Capital [$]',
            xaxis2_title='Count',
            yaxis2_title='Price' if show_candles else '',
            template=theme,
            xaxis_rangeslider_visible=False,
            showlegend=False,
            )

        if cut_off is not None:
            fig_capital.add_annotation(
                x=0.5,
                y=0.5,
                text=f'Wins: {wins}, Losses: {losses}',
                showarrow=False,
                xref='paper',
                yref='paper',
                )
            
        fig_capital.show(title='Capital')

    if oneD_hist:
        # add a vertical histogram of the final capital to the right of the plot
        # use a different color for the bars above initial capital

        fig_1Dhist = go.Figure()
        fig_1Dhist.add_trace(
            go.Histogram(
                y=final_capitals,
                histnorm='percent',
                orientation='h',
                marker=dict(color='rgba(31, 75, 156,1)'),
                # number of bins
                # nbinsx=30,
                ))

        fig_1Dhist.update_traces(
            marker_line_width=0.5, 
            marker_line_color='black',
            # marker=dict(color=['rgba(31, 75, 156, 0.5)' if final_capitals[i] > initial_capital else 'rgba(31, 75, 156, 1)' for i in range(N_test)]),
            )

        # change x axis of the histogram to relative frequency
        fig_1Dhist.update_xaxes(
            title_text='Relative frequency',
            )

        # hide legend
        fig_1Dhist.update_layout(
            showlegend=False,
            template=theme,
            )
        fig_1Dhist.show()

    if twoD_hist:
        fig_2Dhist = px.density_heatmap(
            x=final_N_trades, 
            y=final_capitals, 
            histnorm='percent',
            marginal_x="histogram",
            marginal_y="histogram",
            # number of bins
            # nbinsx=20,
            # nbinsy=100,
            )

        # add individual points to the plot
        if add_individual_trades_on_2Dhist:
            fig_2Dhist.add_trace(
                go.Scatter(
                    x=final_N_trades,
                    y=final_capitals,
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        opacity=0.3,
                        color='black',
                        size=5,
                    ),
                    showlegend=False,))

        fig_2Dhist.update_layout(
            title='Final capital vs number of trades',
            xaxis_title='Number of trades',
            yaxis_title='Final capital [$]',
            template=theme,
            )
        fig_2Dhist.show()

    if indicators is not None:
        # add the indicators to the plots
        fig_indicators = go.Figure()
        fig_indicators.add_trace(
            go.Scatter(
                x=np.arange(len(indicators['ADX'])),
                y=indicators['ADX'],
                mode='lines',
                line=dict(color='black'),
                showlegend=False,
                ))
        fig_indicators.update_layout(
            title='Indicators',
            xaxis_title='Time',
            yaxis_title='Indicator value',
            template=theme,
            )
        fig_indicators.show()

def relative_moving_average(source_data, period=14):

    src = np.asarray(source_data)
    alpha = 1 / period
    result = np.zeros(len(src))

    result[0] = np.nanmean(src[:period])  # equivalent to ta.sma(src, length)
    for i in range(1, len(src)):
        result[i] = alpha * src[i] + (1 - alpha) * result[i - 1]

    return result

def ta_change(arr):
    return np.diff(arr, prepend=np.nan)

def ta_tr(high, low, close):
    return np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))

def dirmov(len, high, low, close):
    ta_rma = relative_moving_average
    up = ta_change(high)
    down = -ta_change(low)
    
    plusDM = np.where(np.isnan(up), np.nan, np.where((up > down) & (up > 0), up, 0))
    minusDM = np.where(np.isnan(down), np.nan, np.where((down > up) & (down > 0), down, 0))
    
    truerange = ta_rma(ta_tr(high, low, close), len)
    plus = 100 * ta_rma(plusDM, len) / truerange
    minus = 100 * ta_rma(minusDM, len) / truerange
    
    return plus, minus

def ADX(close, high, low, adx_period=14):
    ta_rma = relative_moving_average
    plus, minus = dirmov(adx_period, high, low, close)
    sum = plus + minus
    # if sum == 0 replace with 1
    sum = np.where(sum == 0, 1, sum)
    adx = 100 * ta_rma(np.abs(plus - minus) / sum, adx_period)
    return adx

def find_max_prob_profit(out, initial_capital):
    N_test = len(out)
    # find capital with the highest probability of being reached
    capitals = [out[i]['capital'][-1] for i in range(N_test)]
    counts, bin_edges = np.histogram(capitals, bins=30)
    # Find the bin with the highest count
    max_bin_index = np.argmax(counts)
    # Get the bin range (edges) of the bin with the highest count
    max_bin_range = (bin_edges[max_bin_index], bin_edges[max_bin_index + 1])
    capitals_mean = np.mean(capitals)
    # find the bin range with 70% of the counts of the max bin
    std_dev = np.std(capitals)
    region = np.array([capitals_mean- std_dev, capitals_mean+ std_dev])
    region = (region / initial_capital - 1) * 100

    PP1 = (np.mean(max_bin_range)/initial_capital - 1) * 100
    PP2 = (capitals_mean/initial_capital - 1) * 100
    PP = np.array([PP1, PP2])

    return PP, region 

def plot_bounds_with_nans(region, time_range, color, fig):
    import plotly.graph_objects as go
    # highlight the area between the two sides of the region
    # split the reion into parts separated by nan values
    region_sep = np.split(region, np.where(np.isnan(region[:,0]))[0])
    time_range_region = np.split(list(time_range), np.where(np.isnan(region[:,0]))[0])
    for i in range(len(region_sep)):
        if len(region_sep[i]) == 1:
            continue
        fig.add_trace(go.Scatter(
            x=time_range_region[i] if i == 0 else time_range_region[i][1:], 
            y=region_sep[i][:,0] if i == 0 else region_sep[i][1:,0],
            mode='lines', 
            name='Lower Bound', 
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
            ))
        fig.add_trace(go.Scatter(
            x=time_range_region[i] if i == 0 else time_range_region[i][1:],
            y=region_sep[i][:,1] if i == 0 else region_sep[i][1:,1], 
            fill='tonexty',
            mode='none', 
            # change the color of the area
            fillcolor=color,
            showlegend=False,
            # remove hoverinfo
            hoverinfo='skip',
            ))

def get_colormap_colors(cmap_name, N):
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap(cmap_name)
    colors_array = [cmap(i / N) for i in range(N)]  # Get RGBA colors for N evenly spaced values
    # convert to rgba format
    colors = [f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},1)' for color in colors_array]
    # colors opacified
    colors_opac = [f'rgba({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)},0.5)' for color in colors_array]
    return colors, colors_opac

def strategy_zigzag(
        PP, 
        time_range, 
        backward_duration, 
        dt, 
        threshold,
        show_fig=None, 
        fig=None,
        ):
    import plotly.graph_objects as go

    PP_final = [None]*2
    for cntr,cross in enumerate(['cross-below', 'cross-above']):
        status = np.zeros_like(PP, dtype=bool)
        division = int(pd.Timedelta(backward_duration) / dt)
        entries = np.zeros_like(PP, dtype=bool)
        exits = np.zeros_like(PP, dtype=bool)
        i=0
        for _ in time_range:
            if i==0:
                i += 1
                continue
            
            if not status[i]:
                if cross == 'cross-below':
                    cross_cond =PP[i]<=threshold and PP[i-1]>threshold
                elif cross == 'cross-above':
                    cross_cond =PP[i]>=threshold and PP[i-1]<threshold

                if cross_cond:
                    if i+division >= len(PP):
                        break
                    status[i:i+division] = True
                    entries[i] = True
                    exits[i+division] = True
            i += 1

        PP = np.array(PP)
        PP_final[cntr] = np.nansum(PP[exits])
        
        colors = ['green' if value > 0 else 'red' for value in PP[exits]]
        
        if show_fig==cross:
            fig.add_trace(go.Scatter(
                x = time_range[exits],
                y = PP[exits],
                mode = 'markers',
                marker = dict(
                    color = colors,
                    size = 5,
                    ),
                showlegend=False,
                ))
            fig.add_trace(go.Scatter(
                x = time_range[entries],
                y = PP[entries],
                mode = 'markers',
                marker = dict(
                    color = 'black',
                    size = 3,
                    ),
                    showlegend=False,
                ))

            # add the sum of PP on the figure
            fig.add_annotation(
                x=0.5,
                y=1,
                text=f'Sum of PP: {np.nansum(PP[exits]):.2f}%',
                showarrow=False,
                xref='paper',
                yref='paper',
                )
    return PP_final

def GetPriceData(
        symbol, 
        startTime,
        endTime,
        timeframe,
        Nbars,
        duration,
        source = 'MT5',
        indicators_dict = {
            'ATR':      False,
            'ADX':      False,
            'RSI':      False,
        },
        MA_period = 20,
        ):
    
    if source=='MT5':
        # move the hour forward by 2 hours 
        endTime = endTime + pd.DateOffset(hours=2)

        # if Nbars is larger than 99999, get the data in chunks
        rates = pd.DataFrame()  # Initialize an empty DataFrame
        while Nbars > 0:
            Nbars_chunk = min(Nbars, 200000)
            Nbars -= Nbars_chunk

            rates_chunk = mt5.copy_rates_from(
                symbol, 
                ConvertTimeFrametoMT5(timeframe), 
                endTime,
                Nbars_chunk,
            )

            # convert to pandas DataFrame
            rates_chunk = pd.DataFrame(rates_chunk)

            # Add the retrieved chunk to the overall list
            rates = pd.concat([rates, rates_chunk], ignore_index=True)

            # Update endTime to the last time of the retrieved data
            endTime = rates_chunk['time'][0]  # Assuming the data is sorted in reverse chronological order
            
            # convert the endTime from int64 to datetime
            endTime = pd.to_datetime(endTime, unit='s')
            
        # convert times to UTC+1
        rates['time']=pd.to_datetime(rates['time'], unit='s')
        rates['time'] = rates['time'] + pd.DateOffset(hours=-2)

        rates['hour'] = rates['time'].dt.hour

        rates['MA_close'] = rates['close'].rolling(MA_period).mean()
        rates['EMA_close'] = rates['close'].ewm(span=MA_period, adjust=False).mean()

        # remove nans
        rates = rates.dropna()
        rates.rename(columns={'tick_volume': 'volume'}, inplace=True)
        rates['MA_volume'] = rates['volume'].rolling(MA_period).mean()
        rates['EMA_volume'] = rates['volume'].ewm(span=MA_period, adjust=False).mean()
        
        rates['log_volume'] = np.log(rates['volume'])
        rates['MA_log_volume'] = rates['log_volume'].rolling(MA_period).mean()
        rates['EMA_log_volume'] = rates['log_volume'].ewm(span=MA_period, adjust=False).mean()
        
        rates['log_return'] = np.log(rates['close'] / rates['close'].shift(1))
        rates['MA_log_return'] = rates['log_return'].rolling(MA_period).mean()       
        rates['EMA_log_return'] = rates['log_return'].ewm(span=MA_period, adjust=False).mean()
        
        rates['volatility'] = rates['log_return'].rolling(MA_period).std()
        rates['MA_volatility'] = rates['volatility'].rolling(MA_period).std()   
        rates['EMA_volatility'] = rates['volatility'].ewm(span=MA_period, adjust=False).std()
        
        rates['log_volatility'] = np.log(rates['volatility'])
        rates['MA_log_volatility'] = rates['log_volatility'].rolling(MA_period).mean()
        rates['EMA_log_volatility'] = rates['log_volatility'].ewm(span=MA_period, adjust=False).mean()
        
        rates['MA_volume'] = rates['volume'].rolling(MA_period).mean()
        rates['EMA_volume'] = rates['volume'].ewm(span=MA_period, adjust=False).mean()
        
        rates['upward'] = (rates['log_return'] > 0).astype(int)
            

        if indicators_dict['ATR']:
            rates['ATR'] = ta.atr(rates['high'], rates['low'], rates['close'], length=MA_period)
            
        if indicators_dict['ADX']:
            ADX = ta.adx(rates['high'], rates['low'], rates['close'], length=MA_period)
            rates['ADX'] = ADX[f'ADX_{MA_period}']

        if indicators_dict['RSI']:
            rates['RSI'] = ta.rsi(rates['close'], length=MA_period)
      
        return rates
    
    elif source=='yfinance':
        startTime = get_start_time(endTime, timeframe, Nbars)
        # convert the symbol to the format required by yfinance
        # AVAILABLE ASSETS
        # 'USDJPY=X' , 'USDCHF=X' , 'USDCAD=X', 
        # 'EURUSD=X' , 'GBPUSD=X' , 'AUDUSD=X' , 'NZDUSD=X', 
        # 'BTC-USD', 'ETH-USD', 'BNB-USD', 
        # 'XRP-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD'
        if symbol[:3] in ['BTC', 'ETH', 'XRP', 'BNB', 'ADA', 'DOGE', 'DOT', 'SOL']:
            symbol = symbol[:3] + '-' + symbol[3:]
        else:
            symbol = symbol + '=X'
            # pass
        # convert timeframe to yfinance format
        timeframe = ConvertTimeFrametoYfinance(timeframe)
        rates = GetPriceData_Yfinance(symbol, startTime, endTime, timeframe)
        # change keys name from Close, Open, High, Low to close, open, high, low
        rates = rates.rename(columns={'Close':'close', 'Open':'open', 'High':'high', 'Low':'low'})
        # change keys name from Date to time
        rates['time'] = rates.index
        return rates

def ConvertTimeFrametoYfinance(timeframe):
    timeframes = {
        'M1': '1m',
        'M5': '5m',
        'M15': '15m',
        'M30': '30m',
        'H1': '1h',
        'H4': '4h',
        'D1': '1d',
        'W1': '1wk',
        'MN1': '1mo'
    }
    return timeframes.get(timeframe, 'Invalid timeframe')

def ConvertTimeFrametoMT5(timeframe):
    timeframes = {
        'M1': mt5.TIMEFRAME_M1,
        'M2': mt5.TIMEFRAME_M2,
        'M3': mt5.TIMEFRAME_M3,
        'M4': mt5.TIMEFRAME_M4,
        'M5': mt5.TIMEFRAME_M5,
        'M6': mt5.TIMEFRAME_M6,
        'M10': mt5.TIMEFRAME_M10,
        'M12': mt5.TIMEFRAME_M12,
        'M15': mt5.TIMEFRAME_M15,
        'M20': mt5.TIMEFRAME_M20,
        'M30': mt5.TIMEFRAME_M30,
        'H1': mt5.TIMEFRAME_H1,
        'H2': mt5.TIMEFRAME_H2,
        'H3': mt5.TIMEFRAME_H3,
        'H4': mt5.TIMEFRAME_H4,
        'H6': mt5.TIMEFRAME_H6,
        'H8': mt5.TIMEFRAME_H8,
        'H12': mt5.TIMEFRAME_H12,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1
    }
    return timeframes.get(timeframe, 'Invalid timeframe')

def GetPriceData_Yfinance(
        symbol, 
        start_time, 
        end_time, 
        timeframe,
        ):
    import yfinance as yf
    OHLC = yf.Ticker(symbol).history(
                # [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
                interval=timeframe,
                # period=Duration,
                start = start_time,
                end = end_time,
            )
    return OHLC

def get_start_time(
        endTime, 
        timeframe, 
        Nbars,
        ):
    import re
    from datetime import timedelta
    def get_time_per_bar(timeframe):
    # Use regex to capture the numeric part and the unit
        match = re.match(r'([A-Za-z]+)(\d+)', timeframe)
        if not match:
            raise ValueError(f"Invalid timeframe format: {timeframe}")
    
        unit = match.group(1).upper()  # Get the letter part (M, H, D)
        value = int(match.group(2))    # Get the numeric part

        # Convert unit to appropriate timedelta
        if unit == 'M':  # Minutes
            return timedelta(minutes=value)
        elif unit == 'H':  # Hours
            return timedelta(hours=value)
        elif unit == 'D':  # Days
            return timedelta(days=value)
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    # Get time per bar based on the timeframe
    time_per_bar = get_time_per_bar(timeframe)

    # Calculate total time to subtract
    total_time = time_per_bar * Nbars

    # Calculate the startTime
    startTime = endTime - total_time

    return startTime
