from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle

class Trader:

    def run(self, state: TradingState):

        # Orders to be placed on exchange matching engine
        result = {}

        # It will be delivered as TradingState.traderData on next execution.

        print(jsonpickle.encode(state))

        traderData = ""
        # Sample conversion request. Check more details below.
        conversions = 1
        return result, conversions, traderData