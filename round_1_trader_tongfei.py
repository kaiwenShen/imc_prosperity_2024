import copy

import jsonpickle
import numpy as np

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

products = ['AMETHYSTS', 'STARFRUIT']

max_date_back_retrieved = 101 # TODO: under exploration, indetermined

position_limits = [20, 20]


class Trader:
    POSITION_LIMIT = {product: limit for product, limit in zip(products, position_limits)}

    def decode_trader_data(self, state):
        if state.timestamp == 0:
            return []
        return jsonpickle.decode(state.traderData)

    def set_up_cached_trader_data(self, state, traderDataOld):
        # for now we just cache the orderDepth.
        order_depth = state.order_depths['STARFRUIT']
        # so far we cache the mid-price of BBO (best bid and offer)
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        cache_midprice = (int(best_bid) + int(best_ask)) / 2
        cache_imbalance = (best_bid_amount - np.abs(best_ask_amount)) / (best_bid_amount + np.abs(best_ask_amount))
        if state.timestamp == 0:
            return jsonpickle.encode([(cache_midprice, cache_imbalance)])
        new_cache = copy.deepcopy(traderDataOld + [(cache_midprice, cache_imbalance)])
        return jsonpickle.encode(new_cache[-101:]) # take how many data, now is latest 100 data points.

    def update_estimated_position(self, estimated_position, product, amount, side):
        amount = side * abs(amount)
        estimated_position[product] += amount
        return estimated_position

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

    def kevin_acceptable_price_wtb_liquidity_take(self, acceptable_price, product, state, ordered_position,
                                                  estimated_traded_lob, fraction_to_keep=0.15):
        """ same as BBO function,but this function allows to walk the book to take liquidity"""
        order_depth: OrderDepth = state.order_depths[product]
        orders: List[Order] = []
        existing_position = state.position[product] if product in state.position else 0
        buy_available_position = np.round((self.POSITION_LIMIT[product] - existing_position) * (1 - fraction_to_keep))
        sell_available_position = np.round((self.POSITION_LIMIT[product] + existing_position) * (1 - fraction_to_keep))
        if ordered_position[product] > 0:
            # we have long position previously,we need to deduct those from buy
            buy_available_position -= ordered_position[product]
        elif ordered_position[product] < 0:
            # we have short position previously, we need to deduct those from sell
            sell_available_position += ordered_position[product]

        for ask, ask_amount in order_depth.sell_orders.items():
            ask_amount = abs(ask_amount)
            if int(ask) < acceptable_price and buy_available_position > 0:
                # price is good, we check the position limit
                if ask_amount > buy_available_position:
                    # we partially take liquidity
                    ask_amount = buy_available_position
                    estimated_traded_lob[product].sell_orders[ask] += ask_amount  # because lob the amount is negative
                else:
                    # we've eaten the ask
                    estimated_traded_lob[product].sell_orders.pop(ask)
                print("BUY", str(ask_amount) + "x", ask)
                orders.append(Order(product, ask, abs(ask_amount)))
                buy_available_position -= ask_amount
                ordered_position = self.update_estimated_position(ordered_position, product, ask_amount, 1)
        for bid, bid_amount in order_depth.buy_orders.items():
            if int(bid) > acceptable_price and sell_available_position > 0:
                # price is good, we check the position limit
                if bid_amount > sell_available_position:
                    # we adjust the amount to sell
                    bid_amount = sell_available_position
                    estimated_traded_lob[product].buy_orders[bid] -= bid_amount
                else:
                    # we've eaten the bid
                    estimated_traded_lob[product].buy_orders.pop(bid)
                print("SELL", str(bid_amount) + "x", bid)
                orders.append(Order(product, bid, -abs(bid_amount)))
                sell_available_position -= bid_amount
                ordered_position = self.update_estimated_position(ordered_position, product, bid_amount, -1)
        return orders, ordered_position, estimated_traded_lob

    def kevin_residual_market_maker(self, acceptable_price, product, state, ordered_position, estimated_traded_lob,
                                    ):
        orders: List[Order] = []
        existing_position = state.position[product] if product in state.position.keys() else 0
        buy_available_position = self.POSITION_LIMIT[product] - existing_position
        sell_available_position = self.POSITION_LIMIT[product] + existing_position
        if ordered_position[product] > 0:
            # we have long position previously,we need to deduct those from buy
            buy_available_position -= ordered_position[product]
        elif ordered_position[product] < 0:
            # we have short position previously, we need to deduct those from sell
            sell_available_position += ordered_position[product]
        buy_available_position = int(buy_available_position)
        sell_available_position = int(sell_available_position)
        best_estimated_bid, best_estimated_bid_amount = list(estimated_traded_lob[product].buy_orders.items())[0]
        best_estimated_ask, best_estimated_ask_amount = list(estimated_traded_lob[product].sell_orders.items())[0]
        best_estimated_bid, best_estimated_ask = int(best_estimated_bid), int(best_estimated_ask)
        estimated_spread = best_estimated_ask - best_estimated_bid
        limit_buy, limit_sell = 0, 0
        if estimated_spread > 0:
            # it's possible to make a market, without spread it will be market order
            if (best_estimated_ask_amount - 1 > acceptable_price > best_estimated_bid_amount + 1
                    and sell_available_position > 0 and buy_available_position > 0):
                # We can provide liquidity on both side.
                # But we only provide liquidity on the profit max side for simplification.
                # if best_estimated_ask_amount * (
                #         best_estimated_ask - 1 - acceptable_price) > best_estimated_bid_amount * (
                #         acceptable_price - (best_estimated_bid + 1)):
                #     # we provide liquidity by posting selling limit order
                #     limit_sell = 1
                # else:
                #     # we provide liquidity by posting buying limit order
                #     limit_buy = 1
                limit_buy, limit_sell = 1, 1
            elif best_estimated_ask_amount - 1 > acceptable_price and sell_available_position > 0:
                # we provide liquidity by posting selling limit order
                limit_sell = 1
            elif best_estimated_bid_amount + 1 < acceptable_price and buy_available_position > 0:
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

    def kevin_direction_hft(self, predicted_direction, product, state, ordered_position, estimated_traded_lob):
        """
        This function is a high frequency trading strategy that predicts the direction of the mid-price of the product
        if position allowed:
        if predict direction == 1 : we liquidity take the Best ask, and post a limit buy order at the best bid+1
        if predict direction == -1 : we liquidity take the Best bid, and post a limit sell order at the best ask-1
        """
        assert predicted_direction == 1 or predicted_direction == -1
        orders: List[Order] = []
        existing_position = state.position[product] if product in state.position.keys() else 0
        buy_available_position = self.POSITION_LIMIT[product] - existing_position
        sell_available_position = self.POSITION_LIMIT[product] + existing_position
        if ordered_position[product] > 0:
            # we have long position previously,we need to deduct those from buy
            buy_available_position -= ordered_position[product]
        elif ordered_position[product] < 0:
            # we have short position previously, we need to deduct those from sell
            sell_available_position += ordered_position[product]
        order_depth: OrderDepth = state.order_depths[product]
        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
        best_ask, best_bid = int(best_ask), int(best_bid)
        best_ask_amount = abs(best_ask_amount)
        if predicted_direction == 1:
            # we predict the price will go up
            if buy_available_position > 0:
                # we only buy the best ask
                if best_ask_amount > buy_available_position:
                    best_ask_amount = buy_available_position
                    estimated_traded_lob[product].sell_orders[best_ask] += best_ask_amount
                else:
                    # we take the whole best ask
                    estimated_traded_lob[product].sell_orders.pop(best_ask)
                print("BUY", str(best_ask_amount) + "x", best_ask)
                orders.append(Order(product, best_ask, best_ask_amount))
                buy_available_position -= best_ask_amount
                if buy_available_position > 0:
                    # we provide liquidity. note that we must have take the whole best ask so there is definitely a spread
                    print("LIMIT BUY", str(buy_available_position) + "x", best_bid + 1)
                    orders.append(Order(product, best_bid + 1, buy_available_position))
        else:
            # we predict the price will go down
            best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            if sell_available_position > 0:
                # we only sell the best bid
                if best_bid_amount > sell_available_position:
                    best_bid_amount = sell_available_position
                    estimated_traded_lob[product].buy_orders[best_bid] -= best_bid_amount
                else:
                    # we take the whole best bid
                    estimated_traded_lob[product].buy_orders.pop(best_bid)
                print("SELL", str(best_bid_amount) + "x", best_bid)
                orders.append(Order(product, best_bid, -best_bid_amount))
                sell_available_position -= best_bid_amount
                if sell_available_position > 0:
                    # we provide liquidity by selling
                    print("LIMIT SELL", str(sell_available_position) + "x", best_ask - 1)
                    orders.append(Order(product, best_ask - 1, -sell_available_position))
        return orders, ordered_position, estimated_traded_lob

    def kevin_r1_starfruit_pred(self, traderDataOld, state) -> int:
        """
        MODIFY THIS FUNCTION!!!
        :param traderDataOld: use old trade data to compute something
        :param state: you may also need t=0 data
        :return: an int of either 1,0,or -1
        """
        # notice the price parameters are negative, showing reverting
        coef_fit = [1.14536629]
        intercept = 0.02429364

        # coef_fit = [-0.02839707, -0.05048245, -0.10501895, -0.17026582, -0.26236442, -0.37601569, -0.4910231, -0.52319539]
        # intercept = 0.00289805

        # coef_fit = [-0.0219555, -0.03439007, -0.08034013, -0.13276112, -0.2097877, -0.29943137, -0.38010792, -0.35622213, 0.78259484]
        # intercept = 0.01801344
        z = intercept
        z += traderDataOld[-1][1] * coef_fit[0]

        # for i, val in enumerate(traderDataOld[-8:]):
        #     z += val[0] * coef_fit[i]
        # z += traderDataOld[-1][1] * coef_fit[-1]
        calculated_prob = 1 / (1 + np.exp(-z))
        if calculated_prob > 0.6:
            return 1
        elif calculated_prob < 0.4:
            return -1
        else:
            return 0

    def run(self, state: TradingState):
        # read in the previous cache
        traderDataOld = self.decode_trader_data(state)
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
                    10_000, product, state, ordered_position, estimated_traded_lob)
                mm_order, ordered_position, estimated_traded_lob = self.kevin_residual_market_maker(10_000, product,
                                                                                                    state,
                                                                                                    ordered_position,
                                                                                                    estimated_traded_lob)
                result[product] = liquidity_take_order + mm_order
            if product == 'STARFRUIT':
                print(f"TraderDataOld length: {len(traderDataOld)}")
                if len(traderDataOld) > 100:
                    # we have enough data to make prediction
                    predicted_direction = self.kevin_r1_starfruit_pred(traderDataOld, state)
                    print(f"Predicted direction: {predicted_direction}")
                    if predicted_direction != 0:
                        orders, ordered_position, estimated_traded_lob = self.kevin_direction_hft(predicted_direction,
                                                                                                  product, state,
                                                                                                  ordered_position,
                                                                                                  estimated_traded_lob)
                        result[product] = orders
        print('post_trade_position: ' + str(ordered_position))
        # store the new cache
        traderDataNew = self.set_up_cached_trader_data(state, traderDataOld)
        conversions = 1
        return result, conversions, traderDataNew
