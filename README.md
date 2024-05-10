# IMC Prosperity 2 Global Quantitative Trading Challenge

## Introduction
IMC Prosperity 2 is a global quantitative trading challenge hosted by IMC. The competition spans 15 days, divided into five rounds with each round lasting three days, and was extended by two days due to special circumstances. Participants use algorithmic and location-based trading strategies on a virtual island, incorporating skills in Python, Mathematics, Data Science, and Machine Learning, competing to lead their island to prosperity.
## Team Members
- Kevin Shen: [LinkedIn](https://www.linkedin.com/in/kaiwen-shen/)
- Tongfei Zhou: [LinkedIn](https://www.linkedin.com/in/tongfei-zhou/)
- Ziyin Zeng: [LinkedIn](https://www.linkedin.com/in/ziyinzeng1998/)
- Shaoqin Wan: [LinkedIn](https://www.linkedin.com/in/shaoqin-wan/)

## Round 1 
### Introduction
The first round introduced two tradable products: STARFRUIT and AMETHYSTS. AMETHYSTS are historically stable in price, while STARFRUIT's price is more volatile.

### Algorithm challenge
In algorithmic trading, our team analyzed the price volatility of STARFRUIT, finding it linearly related to its prices in previous days. We trained a linear regression model to predict prices and performed market making around these predicted prices for STARFRUIT. For AMETHYSTS, given its historical price stability, we chose to market make around its historical average price.

### Manual challenge
In manual trading, we analyzed the official provided probability distribution of product base prices. We identified two optimal bidding points, 952 and 978, allowing us to purchase SCUBA_GEAR as many as possible and sell it later at a higher price.

The performance gap between teams in the first round was small, indicating that most teams might have adopted similar strategic directions.

## Round 2 
### Introduction
The second round continued with the products from the first round and introduced a new product, ORCHIDS. The value of ORCHIDS is influenced by various observable factors, such as sunlight duration and humidity.

### Algorithm challenge
In the second round, we engaged in arbitrage trading for the newly introduced ORCHIDS. By comparing the price differences between local and external markets, we bought low in the local market and sold high in the external market, profiting from this disparity. We also utilized market price fluctuations to increase our strategy's liquidity at opportune times, further increasing our profits.

### Manual challenge
The manual trading challenge in the second round was identical to a round last year—a simplified version of a foreign exchange arbitrage game involving converting between products to maximize profit. The solution was simple: brute-force all possible pathways within a specified number of exchanges from shells back to shells and their final profit amounts. Of course, if the product count expands to thousands or even millions, advanced search algorithms like BFS or Dijkstra's would be necessary.

## Round 3 
### Introduction
The third round introduced new products: GIFT_BASKET, CHOCOLATE, STRAWBERRIES, and ROSES. GIFT_BASKET consists of the other three products, calculated as 4 * CHOCOLATE + 6 * STRAWBERRIES + 1 * ROSES. Their prices showed significant correlation.

### Algorithm challenge
In the third round of algorithmic trading, our team analyzed the price of GIFT_BASKET. Using a linear regression model, we predicted its fair price and conducted market making based on this prediction. This strategy allowed us to buy and sell based on overbought or oversold signals from our price predictions, utilizing price volatility for profit. Additionally, as the price of GIFT_BASKET was highly correlated with its components—STRAWBERRIES, ROSES, and CHOCOLATE—we also considered trading these components simultaneously to enhance our strategy's profit.

### Manual challenge
In the third round's manual challenge, our team used Monte Carlo simulations to optimize treasure-hunting strategies. By simulating different probability distributions—normal, uniform, Pareto, and Weibull—we assessed potential returns based on other teams' choices. This thorough analysis helped us identify two optimal exploration sites, 85 and 87.

## Round 4 
### Introduction
The fourth round's residents were extremely enthusiastic about coconuts, introducing COCONUT_COUPON. These coupons allowed the purchase of COCONUTS at a specified price at the end of the round, functioning as a tradable commodity.

### Algorithm challenge
In the fourth round's algorithmic trading challenge, our strategy focused on the newly introduced COCONUT_COUPON, effectively a derivative of COCONUTS. By applying the Black-Scholes model, we predicted the theoretical price of COCONUT_COUPON and made trading decisions based on this price. We executed buy or sell orders when the market price of COCONUT_COUPON deviated from our model prediction, leveraging price fluctuations. Additionally, we managed risks through hedging strategies, ensuring our positions were aligned with market conditions to maximize returns. This method allowed us to efficiently use market opportunities while maintaining risk control.

### Manual challenge
In the fourth round's manual trading, we continued to utilize our knowledge of the reserve price distribution for SCUBA_GEAR. We adopted our previously successful strategy for the first bid, while the second bid required considering the average bid of other traders. To optimize our second bidding strategy for SCUBA_GEAR, we initially chose 987 as a safe bid. However, further analysis indicated a low probability of average bids exceeding 990, prompting us to adjust our strategy and ultimately decide on 984 as our second bid.

## Round 5 
### Introduction
The final round did not introduce new products. However, the island's exchange disclosed information about your trading counterparts, which could be used to further optimize your trading algorithms.

### Algorithm challenge
In the final round's algorithmic trading, with no new products introduced, our strategy shifted to analyzing competitors, particularly a standout trader named Rihana. Through detailed analysis of Rihana's profit patterns on various products, we found she excelled particularly in COCONUT, GIFT BASKET, and ROSES. Moreover, her profits on these products were better than those of our strategy.

Based on this discovery, we decided to use Rihana's trading behavior as a signal and mimic her trades on these products. This strategy allowed us to leverage her market intuition, especially on products where she performed best, thus enhancing our trading efficiency and profit potential.

### Manual challenge
The manual trading challenge in the final round involved interpreting news about several virtual companies before the market opened, and players had to predict stock price movements based on the news. Our trading method was to allocate a portfolio with a given number of shells. After thorough discussion, our team decided not to open positions for news reflecting mixed signals or where the upside was limited. For clear positive or negative signals, we maximally weighted our investments. However, we eventually found that while news like potential new market ventures or product quality issues elicited limited market reactions, scandals like severe fraud or a CEO elopement garnered significant market responses.
