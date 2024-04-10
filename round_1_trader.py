import copy

import jsonpickle
import numpy as np

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
        order_depth = state.order_depths['STARFRUIT']
        # so far we cache the mid-price of BBO
        cache = (int(list(order_depth.sell_orders.keys())[0]) + int(list(order_depth.buy_orders.keys())[0])) / 2
        if state.timestamp == 0:
            return jsonpickle.encode([cache])
        new_cache = copy.deepcopy(traderDataOld + [cache])
        return jsonpickle.encode(new_cache[-101:])

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

    def kevin_acceptable_price_liquidity_take(self, acceptable_price, product, state, ordered_position,
                                              estimated_traded_lob):
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

    def kevin_residual_market_maker(self, acceptable_price, product, state, ordered_position, estimated_traded_lob,
                                    fraction=0.5):
        orders: List[Order] = []
        existing_position = state.position[product] if product in state.position.keys() else 0
        buy_available_position = self.POSITION_LIMIT[product] - existing_position
        sell_available_position = self.POSITION_LIMIT[product] + existing_position
        if ordered_position[product] > 0:
            # we have long position previously,we need to deduct those from buy
            buy_available_position = self.POSITION_LIMIT[product] - ordered_position[product]
        elif ordered_position[product] < 0:
            # we have short position previously, we need to deduct those from sell
            sell_available_position = self.POSITION_LIMIT[product] + ordered_position[product]
        buy_available_position = int(buy_available_position * fraction)
        sell_available_position = int(sell_available_position * fraction)
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

    def kevin_r1_starfruit_pred(self, traderDataOld,state) -> int:
        period = 100
        weight = np.array([0.5] + [1] * (period - 1) + [0.5]) / period
        trend = np.convolve(traderDataOld, weight, mode='valid')
        seasonal = [1.0000093043515597, 1.0000211070428724, 1.0000152383125458, 1.000022345699108,
                    0.999981755734476, 1.0000201277219507, 1.000024584608414, 1.0000122806315461,
                    1.0000295094915426, 0.999978299505073, 1.0000082809742898, 1.0000019412627252,
                    0.9999756345423725, 1.000013661647593, 0.9999784554231299, 1.0000202961349598,
                    1.0000298540884995, 1.0000043258289406, 0.999995277221212, 0.9999757052386832,
                    0.9999799667180557, 0.9999906248256922, 0.9999975373519686, 1.0000030863269451,
                    0.9999866804801183, 1.0000081013598336, 1.0000176731098775, 1.00000837486462,
                    1.0000089950669626, 0.9999993078484393, 0.9999989768271729, 0.9999836137414579,
                    1.0000118481323672, 1.0000195575964446, 1.0000116317330972, 1.0000218090384485,
                    1.0000172760570596, 1.0000068110450777, 1.0000298296054153, 1.000033937491367,
                    1.0000180266031455, 1.000023799839715, 1.0000116045953495, 1.0000201098800467,
                    1.000015727500558, 0.9999966065159654, 0.9999963003058516, 1.0000104847307334,
                    0.9999871311481153, 0.9999807483540643, 0.9999655458357679, 0.9999854573709217,
                    0.9999844032175355, 1.0000012280504584, 1.0000082050017551, 0.9999766711282715,
                    1.0000107308199269, 0.9999754461778715, 1.0000048484663144, 0.9999832995548248,
                    0.9999945178551769, 1.0000111378769325, 0.9999993555832961, 0.999987117298801,
                    1.0000029781071635, 1.000021958698426, 0.9999869975787833, 0.9999961361739005,
                    0.9999989320389138, 0.9999786842902914, 1.0000216762291865, 0.9999964819859734,
                    1.0000025774775598, 0.9999865544379073, 0.9999728484518988, 0.9999886673241212,
                    1.0000164763324202, 1.0000149055983432, 1.000004112075749, 1.000012711218479,
                    1.0000009919632946, 0.9999884074639264, 0.9999896848599122, 0.9999824031518577,
                    0.999975165163218, 0.9999807294429381, 0.9999527696199292, 1.0000035392406041,
                    0.9999837850085336, 0.999998435219356, 0.999981602946158, 0.9999933269528802,
                    0.9999929813644886, 0.9999902581745486, 1.000005149521674, 1.000009887563873,
                    0.9999728692009592, 0.9999886729613285, 0.999997420954309, 1.000007092813771,
                    1.0000093043515597,]
        current_time= state.timestamp/100
        seasonal = seasonal[int(current_time % 100)+1]
        return trend*seasonal

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
                liquidity_take_order, ordered_position, estimated_traded_lob = self.kevin_acceptable_price_liquidity_take(
                    10_000, product, state, ordered_position, estimated_traded_lob)
                mm_order, ordered_position, estimated_traded_lob = self.kevin_residual_market_maker(10_000, product,
                                                                                                    state,
                                                                                                    ordered_position,
                                                                                                    estimated_traded_lob)
                result[product] = liquidity_take_order + mm_order
            # if product == 'STARFRUIT':
            #     print(f"TraderDataOld length: {len(traderDataOld)}")
            #     if len(traderDataOld) > 100:
            #         # we have enough data to make prediction
            #         predicted_price = self.kevin_r1_starfruit_pred(traderDataOld,state)
            #         print(f"Predicted price: {predicted_price}")
            #         liquidity_take_order, ordered_position, estimated_traded_lob = self.kevin_acceptable_price_liquidity_take(
            #             predicted_price, product, state, ordered_position, estimated_traded_lob)
            #         mm_order, ordered_position, estimated_traded_lob = self.kevin_residual_market_maker(predicted_price, product,
            #                                                                                             state,
            #                                                                                             ordered_position,
            #                                                                                             estimated_traded_lob)
            #         result[product] = liquidity_take_order
            #     # we dont mm because there is risk of losing money
        print('post_trade_position: ' + str(ordered_position))
        # store the new cache
        traderDataNew = self.set_up_cached_trader_data(state, traderDataOld)
        conversions = 1
        return result, conversions, traderDataNew
