import copy

import jsonpickle
import numpy as np
import pandas as pd

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

products = ['AMETHYSTS', 'STARFRUIT', 'ORCHIDS', 'CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET', 'COCONUT',
            'COCONUT_COUPON']

position_limits = [20, 20, 100, 250, 350, 60, 60, 300, 600]

NUM_OF_DATA_POINT = 10
from logger import Logger
logger = Logger()


class Trader:
    POSITION_LIMIT = {product: limit for product, limit in zip(products, position_limits)}

    @staticmethod
    def decode_trader_data(state):
        if state.timestamp == 0:
            return []
        return jsonpickle.decode(state.traderData)

    @staticmethod
    def extract_from_cache(traderDataNew, product, position):
        """"""
        return [traderDataNew[i][product][position] for i in range(len(traderDataNew))][::-1]

    @staticmethod
    def calculate_mid_price(state, product):
        order_depth = state.order_depths[product]
        best_bid = list(order_depth.buy_orders.keys())[0]
        best_ask = list(order_depth.sell_orders.keys())[0]
        return (best_bid + best_ask) / 2

    @staticmethod
    def stanford_values_extract(order_dict, side=-1):
        tot_vol = 0
        best_val = -1
        mxvol = -1
        for ask, vol in order_dict.items():
            if side == -1:
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        return best_val, tot_vol

    def cal_standford_mid_price_vol(self, state, product):
        buy_price, buy_vol = self.stanford_values_extract(state.order_depths[product].buy_orders, 1)
        sell_price, sell_vol = self.stanford_values_extract(state.order_depths[product].sell_orders, -1)
        return (buy_price + sell_price) / 2, buy_vol + sell_vol

    @staticmethod
    def get_conversion_obs(state, product):
        conversion_data = state.observations.conversionObservations[product]
        sunlight = conversion_data.sunlight
        humidity = conversion_data.humidity
        importTariff = conversion_data.importTariff
        exportTariff = conversion_data.exportTariff
        transportFees = conversion_data.transportFees
        return sunlight, humidity, importTariff, exportTariff, transportFees

    @staticmethod
    def calculate_imbalance(state, product):
        order_depth = state.order_depths[product]
        best_bid_amount = list(order_depth.buy_orders.values())[0]
        best_ask_amount = list(order_depth.sell_orders.values())[0]
        return (best_bid_amount - abs(best_ask_amount)) / (best_bid_amount + abs(best_ask_amount))

    @staticmethod
    def update_estimated_position(estimated_position, product, amount, side):
        amount = side * abs(amount)
        estimated_position[product] += amount
        return estimated_position

    @staticmethod
    def get_best_bid_ask(product, estimated_traded_lob):
        order_depth = estimated_traded_lob[product]
        buy_lob = [price for price in order_depth.buy_orders.keys() if order_depth.buy_orders[price] > 0]
        sell_lob = [price for price in order_depth.sell_orders.keys() if order_depth.sell_orders[price] < 0]
        best_bid = max(buy_lob) if buy_lob else 0
        best_ask = min(sell_lob) if sell_lob else 0
        return best_bid, order_depth.buy_orders.get(best_bid, 0), best_ask, order_depth.sell_orders.get(best_ask, 0)

    @staticmethod
    def get_worst_bid_ask(product, estimated_traded_lob):
        order_depth = estimated_traded_lob[product]
        buy_lob = [price for price in order_depth.buy_orders.keys() if order_depth.buy_orders[price] > 0]
        sell_lob = [price for price in order_depth.sell_orders.keys() if order_depth.sell_orders[price] < 0]
        worst_bid = min(buy_lob) if buy_lob else 0
        worst_ask = max(sell_lob) if sell_lob else 0
        return worst_bid, order_depth.buy_orders.get(worst_bid, 0), worst_ask, order_depth.sell_orders.get(worst_ask, 0)

    @staticmethod
    def r4_current_coconut_fair_price(state):
        order_depth = state.order_depths['COCONUT_COUPON']
        try:
            best_bid = list(order_depth.buy_orders.keys())[0]
        except IndexError:
            # if this time slice the best bid is None, we use the last time slice best bid
            if state.timestamp == 0:
                best_bid = None
            else:
                best_bid = jsonpickle.decode(state.traderData)[-1]['COCONUT'][0]
        try:
            best_ask = list(order_depth.sell_orders.keys())[0]
        except IndexError:
            # if this time slice the best ask is None, we use the last time slice best ask
            if state.timestamp == 0:
                best_ask = None
            else:
                best_ask = jsonpickle.decode(state.traderData)[-1]['COCONUT'][1]
        coconut_best_bid, coconut_best_ask = list(state.order_depths['COCONUT'].buy_orders.keys())[0], list(
            state.order_depths['COCONUT'].sell_orders.keys())[0]
        mid_price_coupon = (best_bid + best_ask) / 2
        intercept = 8841.201392746636
        coef = 1.82459034
        y_t = intercept + coef * mid_price_coupon
        residual = (coconut_best_ask + coconut_best_bid) / 2 - y_t
        return best_bid, best_ask, residual

    @staticmethod
    def r4_current_coconut_coupon_fair_price(state, coupon_best_bid, coupon_best_ask, ):
        order_depth = state.order_depths['COCONUT']
        best_bid = list(order_depth.buy_orders.keys())[0]
        best_ask = list(order_depth.sell_orders.keys())[0]
        mid_price_coconut = (best_bid + best_ask) / 2
        intercept = -4393.50466244335
        coef = 0.50286009
        y_t = intercept + coef * mid_price_coconut
        residual = (coupon_best_ask + coupon_best_bid) / 2 - y_t
        return residual

    def set_up_cached_trader_data(self, state, traderDataOld):
        # for now we just cache the orderDepth.
        # star_midprice = self.calculate_mid_price(state, 'STARFRUIT')
        # star_standford_midprice, star_majority_vol = self.cal_standford_mid_price_vol(state, 'STARFRUIT')
        # star_imbalance = self.calculate_imbalance(state, 'STARFRUIT')
        # orc_midprice = self.calculate_mid_price(state, 'ORCHIDS')
        # orc_standford_midprice, orc_majority_vol = self.cal_standford_mid_price_vol(state, 'ORCHIDS')
        # orc_imbalance = self.calculate_imbalance(state, 'ORCHIDS')
        # sunlight, humidity, importTariff, exportTariff, transportFees = self.get_conversion_obs(state, 'ORCHIDS')
        coupon_best_bid, coupon_best_ask, coconut_residual = self.r4_current_coconut_fair_price(state)
        coconut_coupon_residual = self.r4_current_coconut_coupon_fair_price(state, coupon_best_bid, coupon_best_ask)
        # cache formulation
        current_cache = [{'COCONUT': [coupon_best_bid, coupon_best_ask, coconut_residual, None],
                          'COCONUT_COUPON': [coconut_coupon_residual, None]
                          }]
        # for ORCHIDS, the first four elements are for pure_arb price, conversion_cache, liquidity provide price, liquidity provide amount
        # for COCONUT and COCONUT_COUPON, the last element is for last time slice signal direction.
        if state.timestamp == 0:
            return current_cache
        new_cache = copy.deepcopy(
            traderDataOld + current_cache)
        return new_cache[-NUM_OF_DATA_POINT:]  # take how many data, now is latest 100 data points.

    def cal_available_position(self, product, state, ordered_position):
        existing_position = state.position[product] if product in state.position.keys() else 0
        buy_available_position = self.POSITION_LIMIT[product] - existing_position
        sell_available_position = self.POSITION_LIMIT[product] + existing_position
        if ordered_position[product] > 0:
            # we had a long position previously,we need to deduct those from buy
            buy_available_position -= ordered_position[product]
        elif ordered_position[product] < 0:
            # we have short position previously, we need to deduct those from sell
            sell_available_position += ordered_position[product]
        return buy_available_position, sell_available_position

    def kevin_market_take(self, product, price, amount, available_amount, side, ordered_position, estimated_traded_lob):
        amount = abs(amount)
        if available_amount == 0:
            return [], available_amount, estimated_traded_lob, ordered_position
        if side == 1:
            # we buy the product
            if amount > available_amount:
                amount = available_amount
                estimated_traded_lob[product].sell_orders[price] += amount
            else:
                # we take the whole best ask
                estimated_traded_lob[product].sell_orders.pop(price)
            print("BUY", product, str(amount) + "x", price)
            ordered_position = self.update_estimated_position(ordered_position, product, amount, side)
            available_amount -= amount
            return [Order(product, price, amount)], available_amount, estimated_traded_lob, ordered_position
        else:
            # we sell the product
            if amount > available_amount:
                amount = available_amount
                estimated_traded_lob[product].buy_orders[price] -= amount
            else:
                # we take the whole best bid
                estimated_traded_lob[product].buy_orders.pop(price)
            print("SELL", product, str(amount) + "x", price)
            ordered_position = self.update_estimated_position(ordered_position, product, amount, side)
            available_amount -= amount
            return [Order(product, price, -amount)], available_amount, estimated_traded_lob, ordered_position

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
                # buy price is good, we compute how large an order to submit
                order, buy_available_position, estimated_traded_lob, ordered_position = self.kevin_market_take(
                    product, ask, ask_amount,
                    buy_available_position, 1,
                    ordered_position,
                    estimated_traded_lob)
                orders += order
        for bid, bid_amount in order_depth.buy_orders.items():
            bid = int(bid)
            if bid > acceptable_price and sell_available_position > 0:
                # price is good, we check the position limit
                order, sell_available_position, estimated_traded_lob, ordered_position = self.kevin_market_take(
                    product, bid, bid_amount,
                    sell_available_position, -1,
                    ordered_position,
                    estimated_traded_lob)
                orders += order
        return orders, ordered_position, estimated_traded_lob

    def kevin_residual_market_maker(self, acceptable_price, product, state, ordered_position, estimated_traded_lob,
                                    ):
        orders: List[Order] = []
        buy_available_position, sell_available_position = self.cal_available_position(product, state, ordered_position)
        best_estimated_bid, _, best_estimated_ask, _ = self.get_best_bid_ask(product, estimated_traded_lob)
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
                estimated_traded_lob[product].buy_orders[best_estimated_bid + 1] = buy_available_position
                ordered_position = self.update_estimated_position(ordered_position, product, buy_available_position,
                                                                  1)
            if limit_sell:
                print("LIMIT SELL", str(sell_available_position) + "x", best_estimated_ask - 1)
                orders.append(Order(product, best_estimated_ask - 1, -sell_available_position))
                estimated_traded_lob[product].sell_orders[best_estimated_ask - 1] = -sell_available_position
                ordered_position = self.update_estimated_position(ordered_position, product,
                                                                  -sell_available_position, -1)
        return orders, ordered_position, estimated_traded_lob

    # def kevin_cover_position(self, product, state, ordered_position, estimated_traded_lob):
    #     orders: List[Order] = []
    #     order_depth: OrderDepth = copy.deepcopy(state.order_depths[product])
    #     existing_position = state.position[product] if product in state.position else 0
    #     if existing_position > 0:
    #         # we walk the book to sell
    #         for bid, bid_amount in order_depth.buy_orders.items():
    #             if existing_position != 0:
    #                 order, existing_position, estimated_traded_lob, ordered_position = self.kevin_market_take(
    #                     product, bid, bid_amount,
    #                     existing_position, -1,
    #                     ordered_position,
    #                     estimated_traded_lob)
    #                 orders += order
    #     elif existing_position < 0:
    #         # we walk the book to buy
    #         existing_position = abs(existing_position)
    #         for ask, ask_amount in order_depth.sell_orders.items():
    #             ask_amount = abs(ask_amount)
    #             if existing_position != 0:
    #                 order, existing_position, estimated_traded_lob, ordered_position = self.kevin_market_take(
    #                     product, ask, ask_amount,
    #                     existing_position, 1,
    #                     ordered_position,
    #                     estimated_traded_lob)
    #                 orders += order
    #
    #     return orders, ordered_position, estimated_traded_lob

    # def kevin_direction_hft(self, predicted_direction, product, state, ordered_position, estimated_traded_lob):
    #     """
    #     This function is a high frequency trading strategy that predicts the direction of the mid-price of the product
    #     if position allowed:
    #     if predict direction == 1 : we liquidity take the Best ask, and post a limit buy order at the best bid+1
    #     if predict direction == -1 : we liquidity take the Best bid, and post a limit sell order at the best ask-1
    #     """
    #     assert predicted_direction in [1, -1]
    #     orders: List[Order] = []
    #     buy_available_position, sell_available_position = self.cal_available_position(product, state, ordered_position)
    #     order_depth: OrderDepth = copy.deepcopy(state.order_depths[product])
    #     best_bid, best_bid_amount, best_ask, best_ask_amount = self.get_best_bid_ask(product, estimated_traded_lob)
    #     best_ask_amount = abs(best_ask_amount)
    #     if predicted_direction == 1:
    #         # we predict the price will go up
    #         if buy_available_position > 0:
    #             # # we only buy the best ask if the best ask is within 2 of mid-price
    #             # if best_ask <=np.floor_divide(best_bid+best_ask,2)+1:
    #             #     if best_ask_amount > buy_available_position:
    #             #         best_ask_amount = buy_available_position
    #             #         estimated_traded_lob[product].sell_orders[best_ask] += best_ask_amount
    #             #     else:
    #             #         # we take the whole best ask
    #             #         estimated_traded_lob[product].sell_orders.pop(best_ask)
    #             #     print("BUY", str(best_ask_amount) + "x", best_ask)
    #             #     orders.append(Order(product, best_ask, best_ask_amount))
    #             #     buy_available_position -= best_ask_amount
    #             # if buy_available_position > 0:
    #             print("LIMIT BUY", str(buy_available_position) + "x", best_bid + 1)
    #             orders.append(Order(product, best_bid + 1, buy_available_position))
    #
    #     else:
    #         # we predict the price will go down
    #         if sell_available_position > 0:
    #             # if best_bid >= np.floor_divide(best_bid+best_ask,2)-1:
    #             #     # we only sell the best bid
    #             #     if best_bid_amount > sell_available_position:
    #             #         best_bid_amount = sell_available_position
    #             #         estimated_traded_lob[product].buy_orders[best_bid] -= best_bid_amount
    #             #     else:
    #             #         # we take the whole best bid
    #             #         estimated_traded_lob[product].buy_orders.pop(best_bid)
    #             #     print("SELL", str(best_bid_amount) + "x", best_bid)
    #             #     orders.append(Order(product, best_bid, -best_bid_amount))
    #             #     sell_available_position -= best_bid_amount
    #             # if sell_available_position > 0:
    #             #     # we provide liquidity by selling
    #             print("LIMIT SELL", str(sell_available_position) + "x", best_ask - 1)
    #             orders.append(Order(product, best_ask - 1, -sell_available_position))
    #
    #     return orders, ordered_position, estimated_traded_lob

    def kevin_price_hft(self, predicted_price, product, state, ordered_position, estimated_traded_lob,
                        acceptable_range=2, standford_price=True):
        """
        per stanford logic, we will liquidity take when best bid/ask is within 2 of the predicted price or
        when we have opposite position and the best bid/ask is at +-2 of the predicted price.
        we then provide liquidity if the undercut price is higher/lower than acceptable_ask/acceptable_bid
        """
        orders: List[Order] = []
        buy_available_position, sell_available_position = self.cal_available_position(product, state, ordered_position)
        product_position = state.position[product] if product in state.position.keys() else 0
        product_position += ordered_position[product]
        # we compute liquidity take first
        acceptable_ask = predicted_price + acceptable_range
        acceptable_bid = predicted_price - acceptable_range
        if standford_price:
            best_bid, _ = self.stanford_values_extract(estimated_traded_lob[product].buy_orders, 1)
            best_ask, _ = self.stanford_values_extract(estimated_traded_lob[product].sell_orders, -1)
        else:
            best_bid, _, best_ask, _ = self.get_best_bid_ask(product, estimated_traded_lob)
        print(
            f'price_hft: best_bid: {best_bid}, best_ask: {best_ask}, acceptable_bid: {acceptable_bid}, acceptable_ask: {acceptable_ask}')
        for ask, ask_amount in list(estimated_traded_lob[product].sell_orders.items()):
            if ask <= acceptable_bid or (product_position < 0 and ask == acceptable_bid + 1):
                # we liquidity take the best ask
                if buy_available_position > 0:
                    order, buy_available_position, estimated_traded_lob, ordered_position = self.kevin_market_take(
                        product, ask, abs(ask_amount),
                        buy_available_position, 1,
                        ordered_position,
                        estimated_traded_lob)
                    orders += order
        for bid, bid_amount in list(estimated_traded_lob[product].buy_orders.items()):
            if bid >= acceptable_ask or (product_position > 0 and bid == acceptable_ask - 1):
                # we liquidity take the best bid
                if sell_available_position > 0:
                    order, sell_available_position, estimated_traded_lob, ordered_position = self.kevin_market_take(
                        product, bid, bid_amount,
                        sell_available_position, -1,
                        ordered_position,
                        estimated_traded_lob)
                    orders += order

        # we then decide which way to provide liquidity
        provide_ask, provide_bid = False, False
        if best_ask - 1 >= acceptable_ask and sell_available_position > 0:
            provide_ask = True
        if best_bid + 1 <= acceptable_bid and buy_available_position > 0:
            provide_bid = True

        if provide_ask and provide_bid:
            # we provide liquidity on both side

            print('liquidity provide on both side')
            print("LIMIT SELL", str(sell_available_position) + "x", best_ask - 1)
            orders.append(Order(product, best_ask - 1, -sell_available_position))
            ordered_position = self.update_estimated_position(ordered_position, product, -sell_available_position,
                                                              -1)
            print("LIMIT BUY", str(buy_available_position) + "x", best_bid + 1)
            orders.append(Order(product, best_bid + 1, buy_available_position))
            ordered_position = self.update_estimated_position(ordered_position, product, buy_available_position, 1)
        elif provide_ask:
            print("LIMIT SELL", str(sell_available_position) + "x", best_ask - 1)
            orders.append(Order(product, best_ask - 1, -sell_available_position))
            ordered_position = self.update_estimated_position(ordered_position, product, -sell_available_position, -1)
        elif provide_bid:
            print("LIMIT BUY", str(buy_available_position) + "x", best_bid + 1)
            orders.append(Order(product, best_bid + 1, buy_available_position))
            ordered_position = self.update_estimated_position(ordered_position, product, buy_available_position, 1)

        return orders, ordered_position, estimated_traded_lob

    def shaoqin_r1_starfruit_pred(self, traderDataNew) -> int:
        coef = [0.18898843, 0.20770677, 0.26106908, 0.34176867]
        intercept = 2.356494353223752
        X = np.array([traderDataNew[i]['STARFRUIT'][1] for i in range(len(traderDataNew))])[::-1]
        return int(round(intercept + np.dot(coef, X)))

    def shaoqin_r2_orchids_pred(self, traderDataNew) -> int:
        coef = 1
        intercept = 0
        return int(round(intercept + coef * traderDataNew[-1]['ORCHIDS'][8]))

    def shaoqin_r2_orchids_pred(self, traderDataNew) -> int:
        coef = [0.03505737066667942, 3.7800693377867836, 7.7039004312429835, ]
        intercept = 648.6118462473457
        import_cost = traderDataNew[-1]['ORCHIDS'][6] + traderDataNew[-1]['ORCHIDS'][7] + traderDataNew[-1]['ORCHIDS'][
            8]
        X = np.array([traderDataNew[-1]['ORCHIDS'][4], traderDataNew[-1]['ORCHIDS'][5], import_cost])
        return int(round(intercept + np.dot(coef, X)))

    @staticmethod
    def overhead_calculation(state, product):
        foreign_ask = state.observations.conversionObservations[product].askPrice
        foreign_bid = state.observations.conversionObservations[product].bidPrice
        transport = state.observations.conversionObservations[product].transportFees
        export_tariff = state.observations.conversionObservations[product].exportTariff
        import_tariff = state.observations.conversionObservations[product].importTariff
        fair_bid = foreign_bid - transport - export_tariff - 0.1
        fair_ask = foreign_ask + transport + import_tariff
        return fair_ask, fair_bid

    def kevin_exchange_arb(self, product, state, ordered_position, estimated_traded_lob, traderDataNew,
                           max_limit: int = 100, profit_margin=2):
        # if state.timestamp == 0 and product not in state.position.keys():
        #     conversions = 0
        # else:
        # # we deal with the conversion from last time slice
        # conversions_price, conversions = traderDataNew[-2][product][0], traderDataNew[-2][product][1]
        # # we need to check if our one side liquidity provide is filled or not
        # liquidity_provide_price, liquidity_provide_amount = traderDataNew[-2][product][2], \
        #     traderDataNew[-2][product][3]
        # print(f'cached_conversion_price: {conversions_price}, cached_conversion_vol: {conversions}')
        # print(f"cached_liquidity provide price: {liquidity_provide_price}, "
        #       f"cached_liquidity provide amount: {liquidity_provide_amount}")
        #
        # trade_list = state.own_trades[product] if product in state.own_trades.keys() else []
        # count = 0
        # for trade in trade_list:
        #     print(f"trade: {trade.quantity},{trade.quantity == conversions}")
        #     if trade.price == liquidity_provide_price:
        #         if trade.quantity == abs(conversions):
        #             count += 1
        #         if count>1 or trade.quantity != abs(conversions):
        #             if liquidity_provide_amount > 0:
        #                 conversions-=trade.quantity
        #             else:
        #                 conversions+=trade.quantity
        conversions = -state.position.get(product, 0)
        orders: List[Order] = []
        conversion_price_cache = 0
        conversions_cache = 0
        order_depth: OrderDepth = copy.deepcopy(state.order_depths[product])
        foreign_exchange_ask, foreign_exchange_bid = self.overhead_calculation(state, product)
        print(f"foreign_exchange_ask: {foreign_exchange_ask}, foreign_exchange_bid: {foreign_exchange_bid}")
        buy_available_position, sell_available_position = self.cal_available_position(product, state, ordered_position)
        # calculate available position to use
        buy_available_position = min(buy_available_position, max_limit) + abs(conversions)
        sell_available_position = min(sell_available_position, max_limit) + abs(conversions)
        # Market take the pure arb opportunity
        # flow: we buy at local, sell at foreign
        for ask, ask_amount in order_depth.sell_orders.items():
            if ask < foreign_exchange_bid:
                # we can buy at local, sell at foreign
                if buy_available_position > 0:
                    order, buy_available_position, estimated_traded_lob, ordered_position = self.kevin_market_take(
                        product, ask, ask_amount,
                        buy_available_position, 1,
                        ordered_position,
                        estimated_traded_lob)
                    orders += order
                    ordered_position = self.update_estimated_position(ordered_position, product, 1, 1)
                    conversions_cache += -order[0].quantity  # we want to do the opposite at foreign exchange
                    conversion_price_cache = ask
        # check if there is arb opportunity for buy orderbook
        # flow: we sell at local, buy at foreign
        for bid, bid_amount in order_depth.buy_orders.items():
            if bid > foreign_exchange_ask:
                # we can sell at local, buy at foreign
                if sell_available_position > 0:
                    order, sell_available_position, estimated_traded_lob, ordered_position = self.kevin_market_take(
                        product, bid, bid_amount,
                        sell_available_position, -1,
                        ordered_position,
                        estimated_traded_lob)
                    orders += order
                    ordered_position = self.update_estimated_position(ordered_position, product, 1, -1)
                    conversions_cache += -order[0].quantity  # we want to do the opposite at foreign exchange
                    conversion_price_cache = bid

        # One side liquidity provide arb
        best_bid, best_bid_amount, best_ask, best_ask_amount = self.get_best_bid_ask(product, estimated_traded_lob)
        liquidity_provide_sell = ((best_ask - 1) >= (foreign_exchange_ask + profit_margin)) and (
                sell_available_position > 0)
        liquidity_provide_buy = ((best_bid + 1) <= (foreign_exchange_bid - profit_margin)) and (
                buy_available_position > 0)

        if liquidity_provide_sell and liquidity_provide_buy:
            liquidity_provide_sell = False
        print(f"liquidity_provide_sell: {liquidity_provide_sell}, liquidity_provide_buy: {liquidity_provide_buy}")

        if liquidity_provide_sell:
            liquidity_provide_sell_price = int(round(foreign_exchange_ask + profit_margin))
            print(f"LIMIT SELL, {sell_available_position}x, {liquidity_provide_sell_price}")
            orders.append(Order(product, liquidity_provide_sell_price, -sell_available_position))
            traderDataNew[-1][product][2] = liquidity_provide_sell_price
            traderDataNew[-1][product][3] = -sell_available_position
            ordered_position = self.update_estimated_position(ordered_position, product, -sell_available_position, -1)
            estimated_traded_lob[product].sell_orders[liquidity_provide_sell_price] = -sell_available_position
        if liquidity_provide_buy:
            liquidity_provide_buy_price = int(round(foreign_exchange_bid - profit_margin))
            print(f"LIMIT BUY, {buy_available_position}x, {liquidity_provide_buy_price}")
            orders.append(Order(product, liquidity_provide_buy_price, buy_available_position))
            traderDataNew[-1][product][2] = liquidity_provide_buy_price
            traderDataNew[-1][product][3] = buy_available_position
            ordered_position = self.update_estimated_position(ordered_position, product, buy_available_position, 1)
            estimated_traded_lob[product].buy_orders[liquidity_provide_buy_price] = buy_available_position

        traderDataNew[-1][product][0] = conversion_price_cache
        traderDataNew[-1][product][1] = conversions_cache
        print(f"cached conversions: {conversions_cache}")
        return conversions, orders, ordered_position, estimated_traded_lob, traderDataNew

    def compute_basket_fair_price_deviation(self, state, product):
        eligible_product = ['CHOCOLATE', 'STRAWBERRIES', 'ROSES', 'GIFT_BASKET']
        assert product in eligible_product
        mid_price = {prod: self.calculate_mid_price(state, prod) for prod in eligible_product}
        basket_premium = 379.4904833333333
        if product == 'GIFT_BASKET':
            fair_price = np.dot([4, 6, 1], [mid_price[prod] for prod in eligible_product[:-1]]) + basket_premium
        elif product == 'CHOCOLATE':
            fair_price = mid_price['GIFT_BASKET'] - np.dot([6, 1], [mid_price['STRAWBERRIES'],
                                                                    mid_price['ROSES']]) - basket_premium
            fair_price /= 4
        elif product == 'STRAWBERRIES':
            fair_price = mid_price['GIFT_BASKET'] - np.dot([4, 1], [mid_price['CHOCOLATE'],
                                                                    mid_price['ROSES']]) - basket_premium
            fair_price /= 6
        else:
            # Rose
            fair_price = mid_price['GIFT_BASKET'] - np.dot([4, 6], [mid_price['CHOCOLATE'],
                                                                    mid_price['STRAWBERRIES']]) - basket_premium
        fair_price_deviation = mid_price[product] - fair_price
        return fair_price_deviation, fair_price

    def kevin_spread_trading(self, product, state, ordered_position, estimated_traded_lob,
                             trade_direction, trade_coef,liquidity_fraction: float = 0.3,anchor_product='GIFT_BASKET'):
        orders: List[Order] = []
        worst_bid, worst_bid_amount, worst_ask, worst_ask_amount = self.get_worst_bid_ask(product, estimated_traded_lob)
        buy_available_position, sell_available_position = self.cal_available_position(product, state, ordered_position)
        print('sell_available_position: ' + str(sell_available_position))
        print('buy_available_position: ' + str(buy_available_position))
        if product != anchor_product:
            match_position = abs(state.position.get(anchor_product, 0) * trade_coef - state.position.get(product, 0))
        else:
            match_position = self.POSITION_LIMIT[anchor_product]
        buy_available_position = min(buy_available_position,
                                     match_position)  # we want our component to match the ratio of the basket
        sell_available_position = min(sell_available_position, match_position)
        action = trade_direction * np.sign(trade_coef)
        if action == -1 and sell_available_position > 0:
            # we sell at worst bid, anticipating mean reversion
            # gradually taking position: we take half liquidity of the lob
            liquidity = int(round(sum(estimated_traded_lob[product].buy_orders.values()) * liquidity_fraction))
            pos = min(sell_available_position, liquidity)
            order = Order(product, worst_bid, -pos)
            orders.append(order)
            print(f"SELL {product} {pos}x {worst_bid}")
            ordered_position = self.update_estimated_position(ordered_position, product, -pos,
                                                              -1)
            sell_available_position -= pos

        elif action == 1 and buy_available_position > 0:
            # we buy, anticipating mean reversion
            liquidity = -int(round(sum(estimated_traded_lob[product].sell_orders.values()) * liquidity_fraction))
            pos = min(buy_available_position, liquidity)

            order = Order(product, worst_ask, pos)
            print(f"BUY {product} {pos}x {worst_ask}")
            orders.append(order)
            ordered_position = self.update_estimated_position(ordered_position, product, pos, 1)
        return orders, ordered_position, estimated_traded_lob

    def r4_coconut_signal(self, traderDataNew, k=1, momentum_threshold=0.5):
        residual = self.extract_from_cache(traderDataNew, 'COCONUT', 2)
        residual_momentum = -np.polyfit(np.arange(len(residual)), residual, 1)[0]
        std_residual = 25.490151318439462
        mean_residual = 1.106915685037772e-12
        threshold = mean_residual + k * std_residual
        print(f"coconut residual: {residual[0]}, threshold: {threshold}")
        print(f'coconut residual momentum: {residual_momentum}')
        # if residual[0] > threshold:
        #     if residual_momentum < momentum_threshold:
        #         return -1
        #     else:
        #         return 0
        # elif residual[0] < -threshold:
        #     if residual_momentum > -momentum_threshold:
        #         return 1
        #     else:
        #         return 0
        # else:
        #     return 0
        if residual[0] > threshold:
            return -1, 1 + residual_momentum
        elif residual[0] < -threshold:
            return 1, 1 - residual_momentum
        else:
            return 0, 1

    def r4_coconut_coupon_signal(self, traderDataNew, k=1,):
        residual = self.extract_from_cache(traderDataNew, 'COCONUT_COUPON', 0)
        residual_momentum = -np.polyfit(np.arange(len(residual)), residual, 1)[0]
        mean_residual = 9.503613303725918e-13
        std_residual = 13.381762301052223
        threshold = mean_residual + k * std_residual
        print(f"coconut coupon residual: {residual[0]}, threshold: {threshold}")
        print(f'coconut coupon residual momentum: {residual_momentum}')
        if residual[0] > threshold:
            return -1, 1 + residual_momentum
        elif residual[0] < -threshold:
            return 1, 1 - residual_momentum
        else:
            return 0, 1

    def run(self, state: TradingState):
        # read in the previous cache
        # traderDataOld = self.decode_trader_data(state)
        # calculate this state cache to avoid duplicate calculation
        # traderDataNew = self.set_up_cached_trader_data(state, traderDataOld)
        print(f"position now:{state.position}")
        ordered_position = {product: 0 for product in products}
        estimated_traded_lob = copy.deepcopy(state.order_depths)
        print("Observations: " + str(state.observations))
        print('pre_trade_position: ' + str(state.position))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths.keys():
            # if product == 'AMETHYSTS':
            #     liquidity_take_order, ordered_position, estimated_traded_lob = self.kevin_acceptable_price_wtb_liquidity_take(
            #         10_000, product, state, ordered_position, estimated_traded_lob, limit_to_keep=1)
            #     # result[product] = liquidity_take_order
            #     mm_order, ordered_position, estimated_traded_lob = self.kevin_residual_market_maker(10_000, product,
            #                                                                                         state,
            #                                                                                         ordered_position,
            #                                                                                         estimated_traded_lob)
            #     result[product] = liquidity_take_order + mm_order
            #     # pnl = 3.5k
            # #
            # if product == 'STARFRUIT':
            #     if len(traderDataNew) == NUM_OF_DATA_POINT:
            #         # we have enough data to make prediction
            #         predicted_price = self.shaoqin_r1_starfruit_pred(traderDataNew)
            #         print(f"Predicted price: {predicted_price}")
            #         # cover_orders, ordered_position, estimated_traded_lob = self.kevin_cover_position(product, state,
            #         #                                                                                  ordered_position,
            #         #                                                                                  estimated_traded_lob)
            #         hft_orders, ordered_position, estimated_traded_lob = self.kevin_price_hft(predicted_price,
            #                                                                                   product, state,
            #                                                                                   ordered_position,
            #                                                                                   estimated_traded_lob,
            #                                                                                   acceptable_range=2)
            #         result[product] = hft_orders
            # if product == 'ORCHIDS':
            #     conversions, arb_orders, ordered_position, estimated_traded_lob, traderDataNew = self.kevin_exchange_arb(
            #         product, state,
            #         ordered_position,
            #         estimated_traded_lob,
            #         traderDataNew,
            #         max_limit=5,
            #         profit_margin=1
            #     )
            #
            #     result[product] = arb_orders
            #
            #     print(f"conversions at this time slice: {conversions}")
            # fair_price_deviation, fair_price = self.compute_basket_fair_price_deviation(state, 'GIFT_BASKET')
            # logger.print(f'fair_price_deviation: {fair_price_deviation}, fair_price: {fair_price}')
            # deviation_threshold = 30 / 2  # 25/2
            # if fair_price_deviation > deviation_threshold:
            #     predicted_basket_direction = -1
            # elif fair_price_deviation < -deviation_threshold:
            #     predicted_basket_direction = 1
            # else:
            #     predicted_basket_direction = 0
            # print(f'predicted_basket_direction: {predicted_basket_direction}')
            # if product == 'GIFT_BASKET':
            #
            #     trade_coef = 1
            #
            #     orders, ordered_position, estimated_traded_lob = self.kevin_spread_trading(product, state,
            #                                                                                ordered_position,
            #                                                                                estimated_traded_lob,
            #                                                                                predicted_basket_direction,
            #                                                                                trade_coef)
            #     result[product] = orders
            # if product == 'CHOCOLATE':
            #     trade_coef=-1
            #
            #     orders, ordered_position, estimated_traded_lob = self.kevin_spread_trading(product, state,
            #                                                                                ordered_position,
            #                                                                                estimated_traded_lob,
            #                                                                                predicted_basket_direction,
            #                                                                                trade_coef)
            #     result[product] = orders
            # if product == 'STRAWBERRIES':
            #     trade_coef = -1
            #
            #     orders, ordered_position, estimated_traded_lob = self.kevin_spread_trading(product, state,
            #                                                                                ordered_position,
            #                                                                                estimated_traded_lob,
            #                                                                                predicted_basket_direction,
            #                                                                                trade_coef)
            #     result[product] = orders
            # if product == 'ROSES':
            #     trade_coef = -1
            #
            #     orders, ordered_position, estimated_traded_lob = self.kevin_spread_trading(product, state,
            #                                                                                ordered_position,
            #                                                                                estimated_traded_lob,
            #                                                                                predicted_basket_direction,
            #                                                                                trade_coef)
            #     result[product] = orders

            if product == 'COCONUT':
                if len(traderDataNew) > 9:
                    # we have enough data to make prediction
                    coconut_direction,liquidity_multiplier = self.r4_coconut_signal(traderDataNew)
                else:
                    coconut_direction,liquidity_multiplier = 0,1
                print(f"coconut_direction: {coconut_direction}")
                orders, ordered_position, estimated_traded_lob = self.kevin_spread_trading(product, state,
                                                                                           ordered_position,
                                                                                           estimated_traded_lob,
                                                                                           coconut_direction,
                                                                                           1,
                                                                                           anchor_product='COCONUT',
                                                                                           liquidity_fraction=0.8*liquidity_multiplier)
                result[product] = orders
            if product == 'COCONUT_COUPON':
                if len(traderDataNew) > 9:
                    # we have enough data to make prediction
                    coconut_coupon_direction, liquidity_multiplier = self.r4_coconut_coupon_signal(traderDataNew)
                else:
                    coconut_coupon_direction,liquidity_multiplier = 0,1
                orders, ordered_position, estimated_traded_lob = self.kevin_spread_trading(product, state,
                                                                                           ordered_position,
                                                                                           estimated_traded_lob,
                                                                                           coconut_coupon_direction,
                                                                                           3,
                                                                                           anchor_product='COCONUT',
                                                                                           liquidity_fraction=0.8*liquidity_multiplier)
                result[product] = orders
        conversions = 0
        logger.flush(state, result, conversions, '')
        return result, conversions, jsonpickle.encode(traderDataNew)
        # return result, conversions, ''
