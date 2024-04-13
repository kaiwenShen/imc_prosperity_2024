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

            if product == 'AMETHYSTS':
                if state.timestamp==0:
                    orders.append(Order(product, 1010, 20))
                    orders.append(Order(product, 10, 20))

                if state.timestamp==1:
                    orders.append(Order(product, 1010, 5))
                    orders.append(Order(product, 909, -15))
            result[product] = orders

            # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"

        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData