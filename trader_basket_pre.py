from typing import Dict, List, Tuple, Any
from datamodel import OrderDepth, TradingState, Order
import collections
from collections import defaultdict
import random
import math
import copy
import numpy as np

empty_dict = {'PEARLS': 0, 'BANANAS': 0, 'COCONUTS': 0, 'PINA_COLADAS': 0, 'BERRIES': 0, 'DIVING_GEAR': 0, 'DIP': 0,
              'BAGUETTE': 0, 'UKULELE': 0, 'PICNIC_BASKET': 0}


def def_value():
    return copy.deepcopy(empty_dict)


INF = int(1e9)


class Trader:
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'CHOCOLATE': 250, 'STRAWBERRIES': 350, 'ROSES': 60, 'GIFT_BASKET': 60}
    volume_traded = copy.deepcopy(empty_dict)

    person_position = defaultdict(def_value)
    person_actvalof_position = defaultdict(def_value)

    cont_buy_basket_unfill = 0
    cont_sell_basket_unfill = 0

    basket_std = 25

    def compute_orders_basket(self, order_depth):

        orders = {'GIFT_BASKET': []}
        # prods = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        osell, obuy, best_sell, best_buy, worst_sell, worst_buy, mid_price, vol_buy, vol_sell = {}, {}, {}, {}, {}, {}, {}, {}, {}

        # for p in prods:
        #     osell[p] = collections.OrderedDict(sorted(order_depth[p].sell_orders.items()))
        #     obuy[p] = collections.OrderedDict(sorted(order_depth[p].buy_orders.items(), reverse=True))
        #
        #     best_sell[p] = next(iter(osell[p]))
        #     best_buy[p] = next(iter(obuy[p]))
        #
        #     worst_sell[p] = next(reversed(osell[p]))
        #     worst_buy[p] = next(reversed(obuy[p]))
        #
        #     mid_price[p] = (best_sell[p] + best_buy[p]) / 2
        #     vol_buy[p], vol_sell[p] = 0, 0
        #     for price, vol in obuy[p].items():
        #         vol_buy[p] += vol
        #         if vol_buy[p] >= self.POSITION_LIMIT[p] / 10:
        #             break
        #     for price, vol in osell[p].items():
        #         vol_sell[p] += -vol
        #         if vol_sell[p] >= self.POSITION_LIMIT[p] / 10:
        #             break

        diff_mean_premium = mid_price['GIFT_BASKET'] - mid_price['STRAWBERRIES'] * 6 - mid_price['CHOCOLATE'] * 4 - mid_price[
            'ROSES'] - 379.49

        trade_at = self.basket_std * 0.5

        # pb_pos = self.position.get('GIFT_BASKET', 0)
        # pb_neg = self.position.get('GIFT_BASKET', 0)
        #
        #
        # if self.position['GIFT_BASKET'] == self.POSITION_LIMIT['GIFT_BASKET']:
        #     self.cont_buy_basket_unfill = 0
        # if self.position['GIFT_BASKET'] == -self.POSITION_LIMIT['GIFT_BASKET']:
        #     self.cont_sell_basket_unfill = 0


        if diff_mean_premium > trade_at:
            vol = self.position['GIFT_BASKET'] + self.POSITION_LIMIT['GIFT_BASKET']
            self.cont_buy_basket_unfill = 0  # no need to buy rn
            assert (vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_buy['GIFT_BASKET'], -vol))
                print('BUY'+str(vol)+'@'+str(worst_buy['GIFT_BASKET']))
                self.cont_sell_basket_unfill += 2
                # pb_neg -= vol
        elif diff_mean_premium < -trade_at:
            vol = self.POSITION_LIMIT['GIFT_BASKET'] - self.position['GIFT_BASKET']
            self.cont_sell_basket_unfill = 0  # no need to sell rn
            assert (vol >= 0)
            if vol > 0:
                orders['GIFT_BASKET'].append(Order('GIFT_BASKET', worst_sell['GIFT_BASKET'], vol))
                print('SELL'+str(vol)+'@'+str(worst_sell['GIFT_BASKET']))
                self.cont_buy_basket_unfill += 2
                # pb_pos += vol

        return orders

    def run(self, state: TradingState):
        result = {'GIFT_BASKET': []}

        # Iterate over all the keys (the available products) contained in the order dephts
        self.position['GIFT_BASKET'] = state.position.get('GIFT_BASKET', 0)

        for product in state.market_trades.keys():
            for trade in state.market_trades[product]:
                if trade.buyer == trade.seller:
                    continue
                self.person_position[trade.buyer][product] = 1.5
                self.person_position[trade.seller][product] = -1.5
                self.person_actvalof_position[trade.buyer][product] += trade.quantity
                self.person_actvalof_position[trade.seller][product] += -trade.quantity

        orders = self.compute_orders_basket(state.order_depths)
        result['GIFT_BASKET'] += orders['GIFT_BASKET']

        return result, 1, 'Sample'
