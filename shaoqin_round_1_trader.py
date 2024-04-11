import copy

import jsonpickle
import numpy as np

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

import collections

products = ['AMETHYSTS', 'STARFRUIT']

position_limits = [20, 20]


class Trader:
    POSITION_LIMIT = {product: limit for product, limit in zip(products, position_limits)}
    position = {'STARFRUIT': 0}
    starfruit_cache = []
    starfruit_dim = 4

    def update_estimated_position(self, estimated_position, product, amount, side):
        amount = side * abs(amount)
        estimated_position[product] += amount
        return estimated_position

    def cal_available_position(self, product, state, ordered_position):
        existing_position = state.position[product] if product in state.position.keys() else 0
        buy_available_position = self.POSITION_LIMIT[product] - existing_position
        sell_available_position = self.POSITION_LIMIT[product] + existing_position
        if ordered_position[product] > 0:
            # we have long position previously,we need to deduct those from buy
            buy_available_position -= ordered_position[product]
        elif ordered_position[product] < 0:
            # we have short position previously, we need to deduct those from sell
            sell_available_position += ordered_position[product]
        return buy_available_position, sell_available_position

    def kevin_acceptable_price_BBO_liquidity_take(self, acceptable_price, product, state, ordered_position,
                                                  estimated_traded_lob):
        """
        This function takes the best bid and best ask from the order depth and place a market order to take liquidity
        """
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        existing_position = state.position[product] if product in state.position else 0
        if len(order_depth.sell_orders) != 0:
            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            if int(best_ask) < acceptable_price:
                # we buy the product
                if existing_position + abs(best_ask_amount) > self.POSITION_LIMIT[product]:
                    # adjust buy amount base on limit
                    best_ask_amount = self.POSITION_LIMIT[product] - existing_position  # we max out the position
                else:
                    # we've eaten the best ask
                    estimated_traded_lob[product].sell_orders.pop(best_ask)
                    pass
                if best_ask_amount == 0:
                    # we maxed out our position limit
                    pass
                else:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, abs(best_ask_amount)))
                    ordered_position = self.update_estimated_position(ordered_position, product, best_ask_amount, 1)
        if len(order_depth.buy_orders) != 0:
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            if int(best_bid) > acceptable_price:
                # we sell the product
                # print(f'existing position: {existing_position}, best_bid_amount: {best_bid_amount}')
                # print(f'position limit: {self.POSITION_LIMIT[product]}')
                if existing_position - abs(best_bid_amount) < -self.POSITION_LIMIT[product]:
                    # adjust sell amount base on limit
                    best_bid_amount = existing_position + self.POSITION_LIMIT[product]
                else:
                    # we've eaten the best bid
                    estimated_traded_lob[product].buy_orders.pop(best_bid)
                    pass
                if best_bid_amount == 0:
                    pass
                else:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -abs(best_bid_amount)))
                    ordered_position = self.update_estimated_position(ordered_position, product, best_bid_amount,
                                                                      -1)
        return orders, ordered_position, estimated_traded_lob

    def shaoqin_calc_next_price_starfruit(self):
        coef = [0.18898843, 0.20770677, 0.26106908, 0.34176867]
        intercept = 2.356494353223752
        nxt_price = intercept
        for i, val in enumerate(self.starfruit_cache[-4:]):  # 使用最后四个价格
            nxt_price += val * coef[i]
        return int(round(nxt_price))

    def shaoqin_values_extract(self, order_dict, buy=0):
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

    def shaoqin_compute_orders_regression(self, product, order_depth, acc_bid, acc_ask, LIMIT):
        orders: list[Order] = []

        osell = collections.OrderedDict(sorted(order_depth.sell_orders.items()))
        obuy = collections.OrderedDict(sorted(order_depth.buy_orders.items(), reverse=True))

        sell_vol, best_sell_pr = self.shaoqin_values_extract(osell)
        buy_vol, best_buy_pr = self.shaoqin_values_extract(obuy, 1)

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

    def kevin_acceptable_price_wtb_liquidity_take(self, acceptable_price, product, state, ordered_position,
                                                  estimated_traded_lob, limit_to_keep: int = 1):
        """ same as BBO function,but this function allows to walk the book to take liquidity"""
        order_depth: OrderDepth = copy.deepcopy(state.order_depths[product])
        orders: List[Order] = []
        buy_available_position, sell_available_position = self.cal_available_position(product, state, ordered_position)
        buy_available_position -= limit_to_keep
        sell_available_position -= limit_to_keep
        for ask, ask_amount in order_depth.sell_orders.items():
            ask_amount = abs(ask_amount)
            ask = int(ask)
            if ask < acceptable_price and buy_available_position > 0:
                # price is good, we compute how large an order to submit
                if ask_amount > buy_available_position:
                    # we partially take liquidity
                    ask_amount = buy_available_position
                    estimated_traded_lob[product].sell_orders[ask] += ask_amount  # because lob the amount is negative
                else:
                    # we've eaten the ask
                    estimated_traded_lob[product].sell_orders.pop(ask)
                print("BUY", str(ask_amount) + "x", ask)
                buy_available_position -= ask_amount
                orders.append(Order(product, ask, abs(ask_amount)))
                ordered_position = self.update_estimated_position(ordered_position, product, ask_amount, 1)
        for bid, bid_amount in order_depth.buy_orders.items():
            bid = int(bid)
            if bid > acceptable_price and sell_available_position > 0:
                # price is good, we check the position limit
                if bid_amount > sell_available_position:
                    # we adjust the amount to sell
                    bid_amount = sell_available_position
                    estimated_traded_lob[product].buy_orders[bid] -= bid_amount
                else:
                    # we've eaten the bid
                    estimated_traded_lob[product].buy_orders.pop(bid)
                print("SELL", str(bid_amount) + "x", bid)
                sell_available_position -= bid_amount
                orders.append(Order(product, bid, -abs(bid_amount)))
                ordered_position = self.update_estimated_position(ordered_position, product, bid_amount, -1)
        return orders, ordered_position, estimated_traded_lob

    def kevin_residual_market_maker(self, acceptable_price, product, state, ordered_position, estimated_traded_lob,
                                    ):
        orders: List[Order] = []

        buy_available_position, sell_available_position = self.cal_available_position(product, state, ordered_position)

        best_estimated_bid = max(estimated_traded_lob[product].buy_orders.keys())
        best_estimated_ask = min(estimated_traded_lob[product].sell_orders.keys())
        estimated_spread = best_estimated_ask - best_estimated_bid
        limit_buy, limit_sell = 0, 0
        if estimated_spread > 0:
            # it's possible to make a market, without spread it will be market order
            if (best_estimated_ask - 1 > acceptable_price > best_estimated_bid + 1
                    and sell_available_position > 0 and buy_available_position > 0):
                # We can provide liquidity on both side.
                limit_buy, limit_sell = 1, 1
            elif best_estimated_ask - 1 > acceptable_price and sell_available_position > 0:
                # we provide liquidity by posting selling limit order
                limit_sell = 1
            elif best_estimated_bid + 1 < acceptable_price and buy_available_position > 0:
                # we provide liquidity by posting buying limit order
                limit_buy = 1
            if limit_buy:
                print("LIMIT BUY", str(buy_available_position) + "x", best_estimated_bid + 1)
                orders.append(Order(product, best_estimated_bid + 1, buy_available_position))
                estimated_traded_lob[product].buy_orders[str(best_estimated_bid + 1)] = buy_available_position
                ordered_position = self.update_estimated_position(ordered_position, product, buy_available_position,
                                                                  1)
            if limit_sell:
                print("LIMIT SELL", str(sell_available_position) + "x", best_estimated_ask - 1)
                orders.append(Order(product, best_estimated_ask - 1, -sell_available_position))
                estimated_traded_lob[product].sell_orders[str(best_estimated_ask - 1)] = -sell_available_position
                ordered_position = self.update_estimated_position(ordered_position, product,
                                                                  -sell_available_position, -1)
        return orders, ordered_position, estimated_traded_lob

    def run(self, state: TradingState):
        # read in the previous cache
        print(state.position)
        ordered_position = {product: 0 for product in products}
        estimated_traded_lob = copy.deepcopy(state.order_depths)
        print("Observations: " + str(state.observations))
        print('pre_trade_position: ' + str(state.position))
        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths.keys():
            if product == 'AMETHYSTS':
                # orders that doesn't walk the book
                # liquidity_take_order, ordered_position, estimated_traded_lob = self.kevin_acceptable_price_BBO_liquidity_take(
                #     10_000, product, state, ordered_position, estimated_traded_lob)
                # orders that walk the book
                liquidity_take_order, ordered_position, estimated_traded_lob = self.kevin_acceptable_price_wtb_liquidity_take(
                    10_000, product, state, ordered_position, estimated_traded_lob, limit_to_keep=1)
                # result[product] = liquidity_take_order
                mm_order, ordered_position, estimated_traded_lob = self.kevin_residual_market_maker(10_000, product,
                                                                                                    state,
                                                                                                    ordered_position,
                                                                                                    estimated_traded_lob)
                result[product] = liquidity_take_order + mm_order
            if product == 'STARFRUIT':
                result[product] = []
                self.position[product] = state.position.get(product, 0)

                # 更新STARFRUIT价格缓存
                _, bs_starfruit = self.shaoqin_values_extract(
                    collections.OrderedDict(sorted(state.order_depths[product].sell_orders.items())))
                _, bb_starfruit = self.shaoqin_values_extract(
                    collections.OrderedDict(sorted(state.order_depths[product].buy_orders.items(), reverse=True)),
                    1)
                self.starfruit_cache.append((bs_starfruit + bb_starfruit) / 2)
                if len(self.starfruit_cache) > self.starfruit_dim:
                    self.starfruit_cache.pop(0)

                # 计算可接受的买卖价格
                accept_range = 2
                acc_bid = self.shaoqin_calc_next_price_starfruit() - accept_range
                acc_ask = self.shaoqin_calc_next_price_starfruit() + accept_range

                # 为STARFRUIT生成订单
                order_depth = state.order_depths[product]
                orders = self.shaoqin_compute_orders_regression(product, order_depth, acc_bid, acc_ask, self.POSITION_LIMIT[product])
                result[product] += orders

        print('post_trade_position: ' + str(ordered_position))
        # store the new cache
        conversions = 1
        return result, conversions, 'Can_be_replaced'
