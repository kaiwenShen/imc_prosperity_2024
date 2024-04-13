from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string


class Trader:

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        print("Order depths: " + str(state.order_depths))
        print("Position: " + str(state.position))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            if product == 'AMETHYSTS':
                if product not in state.position.keys() or state.position[product] !=5:
                    orders.append(Order(product, best_ask, max(5,best_ask_amount)))
                else:
                    orders.append(Order(product, best_bid, -5))
                    orders.append(Order(product, best_ask, 10))

            result[product] = orders

            # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData