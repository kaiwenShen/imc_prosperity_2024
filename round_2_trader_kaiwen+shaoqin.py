import copy

import jsonpickle
import numpy as np

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

products = ['AMETHYSTS', 'STARFRUIT', 'ORCHIDS']

position_limits = [20, 20, 100]


class Trader:
    POSITION_LIMIT = {product: limit for product, limit in zip(products, position_limits)}

    def decode_trader_data(self, state):
        if state.timestamp == 0:
            return []
        return jsonpickle.decode(state.traderData)

    def extract_from_cache(self, traderDataOld, product, position):
        return [traderDataOld[i][product][position] for i in range(len(traderDataOld))]

    def set_up_cached_trader_data(self, state, traderDataOld):
        num_of_data_points = 4
        # for now we just cache the orderDepth.
        star_order_depth = state.order_depths['STARFRUIT']
        # so far we cache the mid-price of BBO (best bid and offer)
        star_best_bid, star_best_bid_amount = list(star_order_depth.buy_orders.items())[0]
        star_best_ask, star_best_ask_amount = list(star_order_depth.sell_orders.items())[0]
        star_midprice = (int(star_best_bid) + int(star_best_ask)) / 2
        star_imbalance = (star_best_bid_amount - abs(star_best_ask_amount)) / (
                star_best_bid_amount + abs(star_best_ask_amount))

        orc_order_depth = state.order_depths['ORCHIDS']
        orc_best_bid, orc_best_bid_amount = list(orc_order_depth.buy_orders.items())[0]
        orc_best_ask, orc_best_ask_amount = list(orc_order_depth.sell_orders.items())[0]
        orc_midprice = (int(orc_best_bid) + int(orc_best_ask)) / 2
        orc_imbalance = (orc_best_bid_amount - abs(orc_best_ask_amount)) / (
                orc_best_bid_amount + abs(orc_best_ask_amount))
        if state.timestamp == 0:
            return jsonpickle.encode(
                [{'STARFRUIT': [star_midprice, star_imbalance], 'ORCHIDS': [orc_midprice, orc_imbalance]}])
        new_cache = copy.deepcopy(
            traderDataOld + [{'STARFRUIT': [star_midprice, star_imbalance], 'ORCHIDS': [orc_midprice, orc_imbalance]}])
        return jsonpickle.encode(new_cache[-num_of_data_points:])  # take how many data, now is latest 100 data points.

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

    def get_best_bid_ask(self, product, estimated_traded_lob):
        order_depth = estimated_traded_lob[product]
        buy_lob = [price for price in order_depth.buy_orders.keys() if order_depth.buy_orders[price] > 0]
        sell_lob = [price for price in order_depth.sell_orders.keys() if order_depth.sell_orders[price] < 0]
        best_bid = max(buy_lob) if buy_lob else 0
        best_ask = min(sell_lob) if sell_lob else 0
        return best_bid, order_depth.buy_orders[best_bid], best_ask, order_depth.sell_orders[best_ask]

    def kevin_acceptable_price_BBO_liquidity_take(self, acceptable_price, product, state, ordered_position,
                                                  estimated_traded_lob):
        """
        This function takes the best bid and best ask from the order depth and place a market order to take liquidity
        """
        order_depth: OrderDepth = copy.deepcopy(state.order_depths[product])
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

    def kevin_cover_position(self, product, state, ordered_position, estimated_traded_lob):
        orders: List[Order] = []
        order_depth: OrderDepth = copy.deepcopy(state.order_depths[product])
        existing_position = state.position[product] if product in state.position else 0
        if existing_position > 0:
            # we walk the book to sell
            for bid, bid_amount in order_depth.buy_orders.items():
                if existing_position != 0:
                    if bid_amount > existing_position:
                        bid_amount = existing_position
                        estimated_traded_lob[product].buy_orders[bid] -= bid_amount
                    else:
                        estimated_traded_lob[product].buy_orders.pop(bid)
                    print("Cover SELL", str(bid_amount) + "x", bid)
                    orders.append(Order(product, bid, -bid_amount))
                    ordered_position = self.update_estimated_position(ordered_position, product, bid_amount, -1)
                    existing_position -= bid_amount
        elif existing_position < 0:
            # we walk the book to buy
            existing_position = abs(existing_position)
            for ask, ask_amount in order_depth.sell_orders.items():
                ask_amount = abs(ask_amount)
                if existing_position != 0:
                    if ask_amount > existing_position:
                        ask_amount = existing_position
                        estimated_traded_lob[product].sell_orders[ask] += ask_amount
                    else:
                        estimated_traded_lob[product].sell_orders.pop(ask)
                    print("COVER BUY", str(ask_amount) + "x", ask)
                    orders.append(Order(product, ask, ask_amount))
                    ordered_position = self.update_estimated_position(ordered_position, product, ask_amount, 1)
                    existing_position -= ask_amount

        return orders, ordered_position, estimated_traded_lob

    def kevin_direction_hft(self, predicted_direction, product, state, ordered_position, estimated_traded_lob):
        """
        This function is a high frequency trading strategy that predicts the direction of the mid-price of the product
        if position allowed:
        if predict direction == 1 : we liquidity take the Best ask, and post a limit buy order at the best bid+1
        if predict direction == -1 : we liquidity take the Best bid, and post a limit sell order at the best ask-1
        """
        assert predicted_direction in [1, -1]
        orders: List[Order] = []
        buy_available_position, sell_available_position = self.cal_available_position(product, state, ordered_position)
        order_depth: OrderDepth = copy.deepcopy(state.order_depths[product])
        best_bid, best_bid_amount, best_ask, best_ask_amount = self.get_best_bid_ask(product, estimated_traded_lob)
        best_ask_amount = abs(best_ask_amount)
        if predicted_direction == 1:
            # we predict the price will go up
            if buy_available_position > 0:
                # # we only buy the best ask if the best ask is within 2 of mid-price
                # if best_ask <=np.floor_divide(best_bid+best_ask,2)+1:
                #     if best_ask_amount > buy_available_position:
                #         best_ask_amount = buy_available_position
                #         estimated_traded_lob[product].sell_orders[best_ask] += best_ask_amount
                #     else:
                #         # we take the whole best ask
                #         estimated_traded_lob[product].sell_orders.pop(best_ask)
                #     print("BUY", str(best_ask_amount) + "x", best_ask)
                #     orders.append(Order(product, best_ask, best_ask_amount))
                #     buy_available_position -= best_ask_amount
                # if buy_available_position > 0:
                print("LIMIT BUY", str(buy_available_position) + "x", best_bid + 1)
                orders.append(Order(product, best_bid + 1, buy_available_position))

        else:
            # we predict the price will go down
            if sell_available_position > 0:
                # if best_bid >= np.floor_divide(best_bid+best_ask,2)-1:
                #     # we only sell the best bid
                #     if best_bid_amount > sell_available_position:
                #         best_bid_amount = sell_available_position
                #         estimated_traded_lob[product].buy_orders[best_bid] -= best_bid_amount
                #     else:
                #         # we take the whole best bid
                #         estimated_traded_lob[product].buy_orders.pop(best_bid)
                #     print("SELL", str(best_bid_amount) + "x", best_bid)
                #     orders.append(Order(product, best_bid, -best_bid_amount))
                #     sell_available_position -= best_bid_amount
                # if sell_available_position > 0:
                #     # we provide liquidity by selling
                print("LIMIT SELL", str(sell_available_position) + "x", best_ask - 1)
                orders.append(Order(product, best_ask - 1, -sell_available_position))

        return orders, ordered_position, estimated_traded_lob

    def kevin_price_hft(self, predicted_price, product, state, ordered_position, estimated_traded_lob,
                        acceptable_range=2):
        orders: List[Order] = []
        buy_available_position, sell_available_position = self.cal_available_position(product, state, ordered_position)
        order_depth: OrderDepth = copy.deepcopy(state.order_depths[product])
        best_bid, best_bid_amount, best_ask, best_ask_amount = self.get_best_bid_ask(product, estimated_traded_lob)
        best_ask_amount = abs(best_ask_amount)
        return orders, ordered_position, estimated_traded_lob

    def shaoqin_r1_starfruit_pred(self, traderDataNew) -> int:
        coef = [0.18898843, 0.20770677, 0.26106908, 0.34176867]
        intercept = 2.356494353223752
        return intercept + np.average(self.extract_from_cache(traderDataNew, 'STARFRUIT', 0), weights=coef)

    def run(self, state: TradingState):
        # read in the previous cache
        traderDataOld = self.decode_trader_data(state)
        # calculate this state cache to avoid dulicate calculation
        traderDataNew = self.set_up_cached_trader_data(state, traderDataOld)
        print(f"position now:{state.position}")
        ordered_position = {product: 0 for product in products}
        estimated_traded_lob = copy.deepcopy(state.order_depths)
        print("Observations: " + str(state.observations))
        print('pre_trade_position: ' + str(state.position))
        print('lob starfruit sell: ' + str(state.order_depths['STARFRUIT'].sell_orders))
        print('lob starfruit buy: ' + str(state.order_depths['STARFRUIT'].buy_orders))
        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths.keys():
            # if product == 'AMETHYSTS':
            # # orders that doesn't walk the book
            # # liquidity_take_order, ordered_position, estimated_traded_lob = self.kevin_acceptable_price_BBO_liquidity_take(
            # #     10_000, product, state, ordered_position, estimated_traded_lob)
            # # result[product] = liquidity_take_order
            # # orders that walk the book
            #     liquidity_take_order, ordered_position, estimated_traded_lob = self.kevin_acceptable_price_wtb_liquidity_take(
            #         10_000, product, state, ordered_position, estimated_traded_lob, limit_to_keep=1)
            #     # result[product] = liquidity_take_order
            #     mm_order, ordered_position, estimated_traded_lob = self.kevin_residual_market_maker(10_000, product,
            #                                                                                         state,
            #                                                                                         ordered_position,
            #                                                                                         estimated_traded_lob)
            #     result[product] = liquidity_take_order + mm_order
            if product == 'STARFRUIT':
                #     # if len(traderDataOld) > 100:
                #     # we have enough data to make prediction
                predicted_price = self.shaoqin_r1_starfruit_pred(traderDataNew, state)
                print(f"Predicted direction: {predicted_price}")
                # cover_orders, ordered_position, estimated_traded_lob = self.kevin_cover_position(product, state,
                #                                                                                  ordered_position,
                #                                                                                  estimated_traded_lob)
                cover_orders = []
                hft_orders, ordered_position, estimated_traded_lob = self.kevin_price_hft(predicted_price,
                                                                                          product, state,
                                                                                          ordered_position,
                                                                                          estimated_traded_lob)
                result[product] = cover_orders + hft_orders

        conversions = 1
        return result, conversions, traderDataNew
