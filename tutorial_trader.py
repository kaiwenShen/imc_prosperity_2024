import jsonpickle

import datamodel
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import copy
import string
import numpy as np

products = ['AMETHYSTS', 'STARFRUIT']

position_limits = [20, 20]


class Trader:
    POSITION_LIMIT = {product: limit for product, limit in zip(products, position_limits)}

    def decode_trader_data(self, state):
        if state.timestamp == 0:
            return []
        return jsonpickle.decode(state.traderData)

    def set_up_cached_trader_data(self, state, traderDataOld):
        # for now we just cache the orderDepth.
        cache = state.order_depths.copy()
        if state.timestamp == 0:
            return jsonpickle.encode([cache])
        new_cache = copy.deepcopy(traderDataOld+[cache])
        return jsonpickle.encode(new_cache[-100:])

    def send_market_order(self, product: str, state: datamodel.TradingState, side: int, quantity: int):
        """produce a list of market orders that will walk the order book"""
        # get the existing order depth
        order_depth = state.order_depths[product]
        res = []
        existing_position = state.position[product] if product in state.position else 0
        if side == 1:
            # buy order
            lob = order_depth.sell_orders
            # check if the order broke the limit of the position
            if existing_position + quantity > self.POSITION_LIMIT[product]:
                quantity = self.POSITION_LIMIT[product] - state.position[product]
                if quantity <= 0:
                    return res
        else:
            # sell order
            lob = order_depth.buy_orders
            # check if the order broke the limit of the position
            if existing_position - quantity < -self.POSITION_LIMIT[product]:
                quantity = state.position[product] + self.POSITION_LIMIT[product]
                if quantity <= 0:
                    return res
        remaining_quantity = quantity
        for price, amount in lob.items():
            if remaining_quantity == 0:
                break
            if remaining_quantity < abs(amount):
                # eat part of this level, then our position is full
                res.append(Order(product, int(price), int(-np.sign(amount) * remaining_quantity)))
                remaining_quantity = 0
            else:
                # eat this level entirely
                res.append(Order(product, int(price), -amount))
                remaining_quantity -= abs(amount)
        return res

    def run(self, state: TradingState):
        traderDataOld = self.decode_trader_data(state)
        # Orders to be placed on exchange matching engine
        result = {}
        if state.timestamp == 0:
            for product in products:
                # every product send market order
                result[product] = self.send_market_order(product, state, 1, 10)
        if state.timestamp == 200:
            for product in products:
                # every product send market order
                result[product] = self.send_market_order(product, state, -1, 22)

        traderDataNew = self.set_up_cached_trader_data(state, traderDataOld)

        # Sample conversion request. Check more details below.
        conversions = 0
        # print(jsonpickle.encode(state))
        for product in products:
            print(f'{product} curent_buy_order_depth: {state.order_depths[product].buy_orders}')
            print(f'{product} curent_sell_order_depth: {state.order_depths[product].sell_orders}')
        return result, conversions, traderDataNew
