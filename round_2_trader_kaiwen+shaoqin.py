import copy

import jsonpickle
import numpy as np

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

products = ['AMETHYSTS', 'STARFRUIT', 'ORCHIDS']

position_limits = [20, 20, 100]

NUM_OF_DATA_POINT = 4


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

    def cal_order_majority_mid_price_vol(self, state, product):
        buy_price, buy_vol = self.stanford_values_extract(state.order_depths[product].buy_orders, 1)
        sell_price, sell_vol = self.stanford_values_extract(state.order_depths[product].sell_orders, -1)
        print('cached_number', (buy_price + sell_price) / 2)
        return (buy_price + sell_price) / 2, buy_vol + sell_vol

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
        return best_bid, order_depth.buy_orders[best_bid], best_ask, order_depth.sell_orders[best_ask]

    def set_up_cached_trader_data(self, state, traderDataOld):
        # for now we just cache the orderDepth.
        star_midprice = self.calculate_mid_price(state, 'STARFRUIT')
        star_majority_midprice, star_majority_vol = self.cal_order_majority_mid_price_vol(state, 'STARFRUIT')
        star_imbalance = self.calculate_imbalance(state, 'STARFRUIT')
        orc_midprice = self.calculate_mid_price(state, 'ORCHIDS')
        orc_majority_midprice, orc_majority_vol = self.cal_order_majority_mid_price_vol(state, 'ORCHIDS')
        orc_imbalance = self.calculate_imbalance(state, 'ORCHIDS')
        # cache formulation
        current_cache = [{'STARFRUIT': [star_midprice, star_majority_midprice, star_majority_vol, star_imbalance],
                          'ORCHIDS': [orc_midprice, orc_majority_midprice, orc_majority_vol, orc_imbalance]}]
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
            # we have long position previously,we need to deduct those from buy
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

    def kevin_cover_position(self, product, state, ordered_position, estimated_traded_lob):
        orders: List[Order] = []
        order_depth: OrderDepth = copy.deepcopy(state.order_depths[product])
        existing_position = state.position[product] if product in state.position else 0
        if existing_position > 0:
            # we walk the book to sell
            for bid, bid_amount in order_depth.buy_orders.items():
                if existing_position != 0:
                    order, existing_position, estimated_traded_lob, ordered_position = self.kevin_market_take(
                        product, bid, bid_amount,
                        existing_position, -1,
                        ordered_position,
                        estimated_traded_lob)
                    orders += order
        elif existing_position < 0:
            # we walk the book to buy
            existing_position = abs(existing_position)
            for ask, ask_amount in order_depth.sell_orders.items():
                ask_amount = abs(ask_amount)
                if existing_position != 0:
                    order, existing_position, estimated_traded_lob, ordered_position = self.kevin_market_take(
                        product, ask, ask_amount,
                        existing_position, 1,
                        ordered_position,
                        estimated_traded_lob)
                    orders += order

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
        """
        per stanford logic, we will liquidity take when best bid/ask is within 2 of the predicted price or
        when we have opposite position and the best bid/ask is at +-2 of the predicted price.
        we then provide liquidity if the undercut price is higher/lower than acceptable_ask/acceptable_bid
        """
        orders: List[Order] = []
        buy_available_position, sell_available_position = self.cal_available_position(product, state, ordered_position)
        best_bid,_ = self.stanford_values_extract(estimated_traded_lob[product].buy_orders, 1)
        best_ask,_ = self.stanford_values_extract(estimated_traded_lob[product].sell_orders, -1)
        product_position = state.position[product] if product in state.position.keys() else 0
        product_position += ordered_position[product]
        # we compute liquidity take first
        acceptable_ask = predicted_price + acceptable_range
        acceptable_bid = predicted_price - acceptable_range
        print(f'buy_available_position: {buy_available_position}, sell_available_position: {sell_available_position}')
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

    def run(self, state: TradingState):
        # read in the previous cache
        traderDataOld = self.decode_trader_data(state)
        # calculate this state cache to avoid duplicate calculation
        traderDataNew = self.set_up_cached_trader_data(state, traderDataOld)
        print(f"position now:{state.position}")
        ordered_position = {product: 0 for product in products}
        estimated_traded_lob = copy.deepcopy(state.order_depths)
        print("Observations: " + str(state.observations))
        print('pre_trade_position: ' + str(state.position))
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
                if len(traderDataNew) == NUM_OF_DATA_POINT:
                    # we have enough data to make prediction
                    predicted_price = self.shaoqin_r1_starfruit_pred(traderDataNew)
                    print(f"Predicted price: {predicted_price}")
                    # cover_orders, ordered_position, estimated_traded_lob = self.kevin_cover_position(product, state,
                    #                                                                                  ordered_position,
                    #                                                                                  estimated_traded_lob)
                    print('lob starfruit sell: ' + str(state.order_depths['STARFRUIT'].sell_orders))
                    print('lob starfruit buy: ' + str(state.order_depths['STARFRUIT'].buy_orders))
                    cover_orders = []
                    hft_orders, ordered_position, estimated_traded_lob = self.kevin_price_hft(predicted_price,
                                                                                              product, state,
                                                                                              ordered_position,
                                                                                              estimated_traded_lob)
                    result[product] = cover_orders + hft_orders

        conversions = 0
        return result, conversions, jsonpickle.encode(traderDataNew)
