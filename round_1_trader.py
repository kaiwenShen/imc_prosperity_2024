import copy

import jsonpickle

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

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
        new_cache = copy.deepcopy(traderDataOld + [cache])
        return jsonpickle.encode(new_cache[-100:])

    def official_acceptable_price(self, acceptable_price, product, state):
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        print("Acceptable price : " + str(acceptable_price))
        print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(
            len(order_depth.sell_orders)))

        if len(order_depth.sell_orders) != 0:
            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            if int(best_ask) < acceptable_price:
                print("BUY", str(-best_ask_amount) + "x", best_ask)
                orders.append(Order(product, best_ask, -best_ask_amount))

        if len(order_depth.buy_orders) != 0:
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            if int(best_bid) > acceptable_price:
                print("SELL", str(best_bid_amount) + "x", best_bid)
                orders.append(Order(product, best_bid, -best_bid_amount))
        return orders

    def run(self, state: TradingState):
        # read in the previous cache
        traderDataOld = self.decode_trader_data(state)

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            if product == 'AMETHYSTS':
                result[product] = self.official_acceptable_price(10_000, product, state)
            # String value holding Trader state data required.
        # store the new cache
        traderDataNew = self.set_up_cached_trader_data(state, traderDataOld)
        conversions = 1
        return result, conversions, traderDataNew
