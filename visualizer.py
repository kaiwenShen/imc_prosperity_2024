import pandas as pd
import numpy as np
import glob
import plotly.graph_objects as go


def player_position(player_buy_trade, player_sell_trade, symbol_r_quote):
    player_buy_trade = player_buy_trade.reindex(symbol_r_quote.index).fillna(0)
    player_sell_trade = player_sell_trade.reindex(symbol_r_quote.index).fillna(0)
    player_position = player_buy_trade['quantity'] - player_sell_trade['quantity']
    return player_position.cumsum()


def player_pnl(player_buy_trade, player_sell_trade, symbol_r_quote):
    player_buy_trade = player_buy_trade.reindex(symbol_r_quote.index).fillna(0)
    player_sell_trade = player_sell_trade.reindex(symbol_r_quote.index).fillna(0)
    player_capital = (player_sell_trade['quantity'] * player_sell_trade['price']).cumsum() - (
                player_buy_trade['quantity'] * player_buy_trade['price']).cumsum()
    player_market_value = player_position(player_buy_trade, player_sell_trade, symbol_r_quote) * (
                (symbol_r_quote['bid_price_1'] + symbol_r_quote['ask_price_1']) / 2)
    player_pnl = player_capital + player_market_value
    return player_pnl


def metrics(pnl):
    sharpe = pnl.iloc[-1] / np.std(pnl)
    print(f'Sharpe: {sharpe:,.2f}')
    print(f'PnL: {pnl.iloc[-1]}')
    print(f'Zen Score: {pnl.iloc[-1] ** 0.4 * sharpe ** 0.6:,.2f}')


def visualize_player_product(player, product, round,plot=True):
    # get all files under src/round1
    files = glob.glob('src/round5/round-5-island-data-bottle/*.csv')
    round_files = glob.glob(f'src/round{round}/round-{round}-island-data-bottle/*.csv')
    r_quote = []
    for file in round_files:
        if 'prices' in file:
            print(file)
            df = pd.read_csv(file, sep=';', index_col=0)
            r_quote.append(df)
    r_quote = pd.concat(r_quote).reset_index().set_index(['day', 'timestamp']).sort_index().reset_index()

    r_quote['merged_timestamp'] = (r_quote['day'] + 2) * 1000000 + r_quote['timestamp']

    r_quote.set_index('merged_timestamp', inplace=True)

    r = []
    df_keys = []
    for file in files:
        if f'round_{round}' in file:
            print(file)
            df = pd.read_csv(file, sep=';', index_col=0)
            r.append(df)
            df_keys.append(int(file.split(f'trades_round_{round}_day_')[1].split('_wn.csv')[0]))
    r = pd.concat(r, keys=df_keys, names=['day']).reset_index().set_index(
        ['day', 'timestamp']).sort_index().reset_index()

    print(
        f'player: {[x for x in r[r['symbol'] == product]['buyer'].unique() if x in r[r['symbol'] == product]['seller'].unique()]}')

    r['merged_timestamp'] = (r['day'] + 2) * 1000000 + r['timestamp']
    r.set_index('merged_timestamp', inplace=True)

    # all trade in r where buyer is player or seller is player
    tmp = r[r['symbol'] == product]
    player_buy_trade = tmp[tmp['buyer'] == player]
    player_sell_trade = tmp[tmp['seller'] == player]

    player_buy_trade = player_buy_trade.groupby(level=0).agg({'price': 'mean',
                                                              'quantity': 'sum'})
    player_sell_trade = player_sell_trade.groupby(level=0).agg({'price': 'mean',
                                                                'quantity': 'sum'})



    r_prod_quote = r_quote[r_quote['product'] == product]
    position = player_position(player_buy_trade, player_sell_trade, r_prod_quote)
    pnl = player_pnl(player_buy_trade, player_sell_trade, r_prod_quote)
    if plot:
        # Create a figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=r_prod_quote.index,
            y=r_prod_quote['bid_price_1'],
            mode='lines',
            line=dict(width=1, color='blue'),
            name='Bid Price',
            yaxis='y'
        ))

        fig.add_trace(go.Scatter(
            x=r_prod_quote.index,
            y=r_prod_quote['ask_price_1'],
            mode='lines',
            line=dict(width=1, color='purple'),
            name='Ask Price',
            yaxis='y'
        ))

        # Scatter plot for buy trades
        fig.add_trace(go.Scatter(
            x=r_prod_quote.index,
            y=player_buy_trade.reindex(r_prod_quote.index)['price'],
            mode='markers',
            marker=dict(
                size=player_buy_trade.reindex(r_prod_quote.index)['quantity'].fillna(0) * 2,
                color='red'
            ),
            name='Buy Trades',
            yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=r_prod_quote.index,
            y=player_sell_trade.reindex(r_prod_quote.index)['price'],
            mode='markers',
            marker=dict(
                size=player_sell_trade.reindex(r_prod_quote.index)['quantity'].fillna(0) * 2,
                color='green'
            ),
            name='Sell Trades',
            yaxis='y'
        ))

        fig.add_trace(go.Scatter(
            x=r_prod_quote.index,
            y=position,
            mode='lines',
            line=dict(width=1, color='orange'),
            name='Position',
            yaxis='y3'
        ))

        fig.add_trace(go.Scatter(
            x=r_prod_quote.index,
            y=pnl,
            mode='lines',
            line=dict(width=1, color='yellow'),
            name='PnL',
            yaxis='y2'
        ))

        # Update layout for a better look
        fig.update_layout(
            title=f'{player} Trading {product} in Round {round}',
            xaxis_title='Time',
            legend_title='Legend',
            template='plotly_dark',  # Optional: Use a dark theme for the plot
            yaxis=dict(title="Price", title_standoff=0.5),
            yaxis2=dict(
                position=0.95,
                title='PnL',
                overlaying='y',
                anchor='free',
                side='right'
            ),
            yaxis3=dict(
                title='Position',
                position=0.05,
                anchor="free",
                overlaying="y",
                side="left",
                title_standoff=0.5,
            ),
            legend=dict(
                title='Legend',
                x=1.2,  # Moves the legend to the far right
                xanchor='right',  # Anchors the legend at the right edge
                y=1,  # Positions the top of the legend at the top of the plot
                yanchor='top'  # Anchors the legend's top at y position
            ),
            height=600
        )

        # Show the plot
        fig.show()
        return position, pnl,r_prod_quote,player_buy_trade,player_sell_trade