from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string


class Trader:

    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        print("Position: " + str(state.position))
        result = {}
        # Orders to be placed on exchange matching engine
        best_ask, best_ask_amount = list(state.order_depths['ORCHIDS'].sell_orders.items())[0]

        result = {'ORCHIDS': [Order('ORCHIDS', best_ask, abs(best_ask_amount))]}


            # String value holding Trader state data required.
        # It will be delivered as TradingState.traderData on next execution.
        traderData = "SAMPLE"
        ask = state.observations.conversionObservations['ORCHIDS'].askPrice
        # Sample conversion request. Check more details below.
        # if 'ORCHIDS' in state.position.keys():
        #     conversions = -state.position['ORCHIDS']
        # else:
        #     conversions = 0
        print(f'market trade on Orchids: {state.market_trades["ORCHIDS"] if "ORCHIDS" in state.market_trades.keys() else None}')
        print(f'own trades on Orchids: {state.own_trades["ORCHIDS"] if "ORCHIDS" in state.own_trades.keys() else None}')
        conversions=-1
        return result, conversions, traderData