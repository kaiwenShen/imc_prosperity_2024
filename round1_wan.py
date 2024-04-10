from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import collections
import numpy as np


class Trader:
    # 初始化STARFRUIT相关的属性
    position = {'STARFRUIT': 0}
    POSITION_LIMIT = {'STARFRUIT': 20}
    starfruit_cache = []
    starfruit_dim = 4

    def __init__(self):
        self.starfruit_cache = []

    def calc_next_price_starfruit(self):
        coef = [0.18898843, 0.20770677, 0.26106908, 0.34176867]
        intercept = 2.356494353223752
        nxt_price = intercept
        for i, val in enumerate(self.starfruit_cache[-4:]):  # 使用最后四个价格
            nxt_price += val * coef[i]
        return int(round(nxt_price))

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1
        for ask, vol in order_dict.items():
            if buy == 0:
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        return tot_vol, best_val

    # def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
    #     orders = []
    #
    #     osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
    #     obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))
    #
    #     sell_vol, best_sell_pr = self.values_extract(osell)
    #     buy_vol, best_buy_pr = self.values_extract(obuy, 1)
    #     cpos = self.position[product]
    #
    #     bid_pr = best_buy_pr + 1
    #     sell_pr = best_sell_pr - 1
    #
    #     if cpos < LIMIT:
    #         num = LIMIT - cpos
    #         orders.append(Order(product, bid_pr, num))
    #     if cpos > -LIMIT:
    #         num = -LIMIT - cpos
    #         orders.append(Order(product, sell_pr, num))
    #
    #     return orders

    def compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.values_extract(osell)
        buy_vol, best_buy_pr = self.values_extract(obuy, 1)

        cpos = self.position[product]

        for ask, vol in osell.items():
            if ((ask <= acc_bid) or ((self.position[product] < 0) and (ask == acc_bid + 1))) and cpos < LIMIT:
                order_for = min(-vol, LIMIT - cpos)
                cpos += order_for
                assert (order_for >= 0)
                orders.append(Order(product, ask, order_for))

        undercut_buy = best_buy_pr + 1
        undercut_sell = best_sell_pr - 1

        bid_pr = min(undercut_buy, acc_bid)  # we will shift this by 1 to beat this price
        sell_pr = max(undercut_sell, acc_ask)

        if cpos < LIMIT:
            num = LIMIT - cpos
            orders.append(Order(product, bid_pr, num))
            cpos += num

        cpos = self.position[product]

        for bid, vol in obuy.items():
            if ((bid >= acc_ask) or ((self.position[product] > 0) and (bid + 1 == acc_ask))) and cpos > -LIMIT:
                order_for = max(-vol, -LIMIT - cpos)
                # order_for is a negative number denoting how much we will sell
                cpos += order_for
                assert (order_for <= 0)
                orders.append(Order(product, bid, order_for))

        if cpos > -LIMIT:
            num = -LIMIT - cpos
            orders.append(Order(product, sell_pr, num))
            cpos += num

        return orders

    def compute_orders(self, product, order_depth, acc_bid, acc_ask):
        if product == "STARFRUIT":
            return self.compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)  # 使用state中的traderData
        print("Observations: " + str(state.observations))  # 打印观察值

        result = {'STARFRUIT': []}
        self.position['STARFRUIT'] = state.position.get('STARFRUIT', 0)

        # 更新STARFRUIT价格缓存
        _, bs_starfruit = self.values_extract(
            collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].sell_orders.items())))
        _, bb_starfruit = self.values_extract(
            collections.OrderedDict(sorted(state.order_depths['STARFRUIT'].buy_orders.items(), reverse=True)), 1)
        self.starfruit_cache.append((bs_starfruit + bb_starfruit) / 2)
        if len(self.starfruit_cache) > self.starfruit_dim:
            self.starfruit_cache.pop(0)

        # 计算可接受的买卖价格
        accept_range = 2
        acc_bid = self.calc_next_price_starfruit() - accept_range
        acc_ask = self.calc_next_price_starfruit() + accept_range

        # 为STARFRUIT生成订单
        order_depth = state.order_depths['STARFRUIT']
        orders = self.compute_orders('STARFRUIT', order_depth, acc_bid, acc_ask)
        result['STARFRUIT'] += orders

        traderData = "SAMPLE"  # 示例traderData，实际应用中需要根据需要进行序列化和更新
        conversions = 1  # 示例转换请求，实际应用中需要根据具体逻辑进行调整

        return result, conversions, traderData  # 返回订单结果、转换请求和更新后的traderData
