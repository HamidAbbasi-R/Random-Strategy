#%%
from plotly import graph_objects as go
import numpy as np
import pandas as pd
from modules import plot_bounds_with_nans
from modules import get_colormap_colors
from modules import strategy_zigzag
assets = ['BNB-USD']
add_bounds = True
add_strategy = True

colors, colors_opac = get_colormap_colors('tab10', len(assets))
fig = go.Figure()
# fig_cumulative = go.Figure()
for i,asset in enumerate(assets):
    out = np.load(f'./results/{asset}_results.npy', allow_pickle=True)
    time_range = out.item()['time_range']
    results = out.item()['results']

    PP1 = [results[k][0][0] for k in range(len(results))]
    PP2 = [results[k][0][1] for k in range(len(results))]
    PP = PP2
    region = np.array([results[i][1] for i in range(len(results))])

    fig.add_trace(go.Scatter(
        x=time_range, 
        y=PP,
        mode='lines',
        line_color=colors[i],
        name=f'{asset}',
        showlegend=True,
        ))

    if add_bounds:
        plot_bounds_with_nans(region, time_range, colors_opac[i], fig)

    # show cumulative profit
    # cum_profit = np.nancumsum(PP)
    # cum_profit_upper = np.nancumsum(region[:,1])
    # cum_profit_lower = np.nancumsum(region[:,0])
    
    # fig_cumulative.add_trace(go.Scatter(
    #     x=time_range, 
    #     y=cum_profit_upper,
    #     mode='lines',
    #     line_color=colors[i],
    #     line=dict(dash='dash', width=0.5),
    #     name=f'{asset} (cumulative upper bound)',
    #     showlegend=False,
    #     ))
    # fig_cumulative.add_trace(go.Scatter(
    #     x=time_range, 
    #     y=cum_profit,
    #     mode='lines',
    #     line_color=colors[i],
    #     fill='tonexty',
    #     name=f'{asset} (cumulative)',
    #     showlegend=True,
    #     ))
    # fig_cumulative.add_trace(go.Scatter(    
    #     x=time_range, 
    #     y=cum_profit_lower,
    #     mode='lines',
    #     line_color=colors[i],
    #     line=dict(dash='dash', width=0.5),
    #     fill='tonexty',
    #     name=f'{asset} (cumulative lower bound)',
    #     showlegend=False,
    #     ))
    

# add line y=0
# fig.add_trace(go.Scatter(
#     x=time_range, 
#     y=[0]*len(PP1),
#     mode='lines',
#     line=dict(dash='dash'),
#     line_color='black',
#     # remove legend
#     showlegend=False,
#     ))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Profit (%)',
    template='ggplot2',
    legend=dict(
        yanchor="top",
        y=1.2,
        xanchor="left",
        x=0,
        # horizontal legend
        orientation="h",
        ),
    )
fig.update_yaxes(
    zeroline=True, 
    zerolinewidth=2, 
    zerolinecolor='black',
    )

# fig_cumulative.update_layout(
#     xaxis_title='Date',
#     yaxis_title='Cumulative Profit (%)',
#     template='ggplot2',
#     legend=dict(
#         yanchor="top",
#         y=1.2,
#         xanchor="left",
#         x=0,
#         orientation="h",
#         ),
#     )
# fig_cumulative.update_yaxes(
#     zeroline=True, 
#     zerolinewidth=2, 
#     zerolinecolor='black',
#     )

# fig_cumulative.show()
fig.show()


if add_strategy:
    threhsold_range = np.arange(np.nanmax(PP), np.nanmin(PP), -0.01)
    # threhsold_range = [0.0]
    backward_duration = out.item()['backward_simulation_duration']
    dt = pd.to_datetime(time_range[1]) - pd.to_datetime(time_range[0])
    PP_final = [[None,None]]*len(threhsold_range)
    for i,threshold in enumerate(threhsold_range):
        PP_final[i] = strategy_zigzag(    
            PP, 
            time_range, 
            backward_duration, 
            dt, 
            threshold,
            show_fig=None if len(threhsold_range)>1 else 'cross-below',
            fig = fig,
            )
        
    fig_PP = go.Figure()
    # colors = ['green' if value > 0 else 'red' for value in PP_final]
    PP_final = np.array(PP_final)
    fig_PP.add_trace(go.Scatter(
        x=threhsold_range, 
        y=PP_final[:,0],
        mode='markers+lines',
        marker = dict(
            size = 2,
            ),
        line=dict(
            color = 'black',
            ),
        name='cross below',
        ))
    fig_PP.add_trace(go.Scatter(
        x=threhsold_range, 
        y=PP_final[:,1],
        mode='markers+lines',
        marker = dict(
            size = 2,
            ),
        line=dict(
            color = 'blue',
            ),
        name='cross above',
        ))
    
    fig_PP.update_layout(
        xaxis_title='Trigger Profit (%)',
        yaxis_title='Overall Profit (%)',
        template='ggplot2',
        legend=dict(
            yanchor="top",
            y=1.2,
            xanchor="left",
            x=0,
            orientation="h",
            ),
        )

    PP_final_max_below = PP_final[:,0].max()
    PP_final_max_above = PP_final[:,1].max()
    PP_final_max_idx_below = PP_final[:,0].argmax()
    PP_final_max_idx_above = PP_final[:,1].argmax()
    colors = ['black', 'blue']
    fig_PP.add_trace(go.Scatter(
        x=[threhsold_range[PP_final_max_idx_below], threhsold_range[PP_final_max_idx_above]], 
        y=[PP_final_max_below, PP_final_max_above],
        mode='markers',
        marker=dict(size=10, color=colors),
        showlegend=False,
        ))
    
    fig_PP.update_yaxes(
        zeroline = True,
        zerolinewidth = 2,
        zerolinecolor = 'black',

    )
    # legend on top


    fig_PP.show()

