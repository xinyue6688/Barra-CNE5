# Code Manual 代码使用说明

## 1. Barra-CNE5.pdf
- This document provides a reference for the Barra description of style factors in the CNE5 model.

## 2. barra_cne5_factor.py

- Update style factor exposure for each stock to todays's date.
- 注意：最近一次更新的数据为8.27，为了避免因子暴露数据被错误的更新覆盖的可能性，当前代码中，更新后的因子暴露落在数据库的位置与历史数据不同，后期调试后如需要正确的迭代，设置为同一路径即可。
  
### Style Factor Construction 风格因子构建

*All style factors are winsorized and standardized.*

#### Beta Factor 市场暴露因子:
- **Calculated through regression analysis to derive the Beta and Alpha values for each stock.**

- **Regression Model:  $\ r_t - r_{f_t} = \alpha + \beta R_t + e_t \$**  
    Where:
    - $\ r_t \$: Stock return at time t
    - $\ r_{f_t} \$: Risk-free return
    - $\ R_t \$: Market return at time t
    - $\ \alpha \$: Excess return
    - $\ \beta \$: Market exposure (Beta value)
    - $\ e_t \$: Error term

- **ALPHA**: Represents the excess return achieved by the stock after accounting for market returns[^1], derived from the regression analysis. It reflects whether the stock's performance exceeds expectations.
- **BETA**: Measures the relationship between the stock return and the market return, indicating the stock's exposure to market movements. Regressed using 252 window with half-life 63.
- **SIGMA**: Represents the difference between the stock return and the risk-free return, adjusted for market returns, reflecting the impact of market risk on stock performance.

[^1]: CSI All Index 000985.CSI

#### Momentum Factor 动量因子:

- **$RSTR = \sum_{t=1}^{504}  \log(1 + r_t) - \log(1 + r_{f_t})$**
    - Relative Strength Factor, calculated as the sum of excess log returns over a trailing period of T = 504 trading days with a lag of L = 21 trading days.
    - Where:
      - $\ r_t \$: Stock return at time t
      - $\ r_{f_t} \$: Risk-free return at time t

#### Size Factor 市值因子

- **LNCAP: Natural log of market cap**

#### Earnings Yield 收益因子

- **EARNYILD  = 0.68 · EPIBS + 0.11 · ETOP + 0.21 · CETOP**

  - **EPIBS**: Earnings-to-price ratio forecasted by analysts[^2] (same year or 1 year forecast).
  - **ETOP**: Computed by dividing the trailing 12-month earnings[^3] by the current market capitalization.
  - **CETOP**: Computed by dividing the trailing 12-month cash earnings[^4] divided by current price.

[^2]: EST_PE in ASHARECONSENSUSROLLINGDATA
[^3]: NET_PROFIT_TTM in ASHARETTMHIS
[^4]: NET_INCR_CASH_CASH_EQU_TTM in ASHARETTMHIS

#### Residual Volatility 残差波动率因子

- **RESVOL = 0.74· DASTD + 0.16 · CMRA + 0.10 · HSIGMA**
  
  - **DASTD**: Computed as the volatility of daily excess returns over the past 252 trading days with a half-life of 42 trading days
  - **CMRA**: Calculative by the maximum cumulative excess returns subtracted by minimum cumulative excess returns
  - **HSIGMA**: Trailing 252 days volatility with half-life 63 days of residual term in BETA factor regression, then orthogonalized with BETA.


#### Growth Factor 成长因子: 
- **GROWTH = 0.47 · SGRO + 0.24 · EGRO + 0.18 · EGIBS + 0.11 · EGIBS_s**

  - **SGRO**: 5 years of sales growth[^5] regressed with years = 1, 2, 3, 4, 5
  - **EGRO**: 5 years of earnings per share growth[^6] regressed with years = 1, 2, 3, 4, 5
  - **EGIBS**: NEED DATA
  - **EGIBS_s**: NEED DATA

*Constructed using the first two sub-factors due to data limitations, since the weights of the last two sub-factors are relatively smaller.*

[^5]: S_FA_GRPS in ASHAREFINANCIALINDICATOR  
[^6]: S_FA_EPS_BASIC in ASHAREFINANCIALINDICATOR

#### Book-to-price 价值因子：
- **BTOP: Last reported book value of common equity divided by current market capitalization[^7]**

[^7]: S_VAL_PB_NEW in ASHAREEODDERIVATIVEINDICATOR

#### Leverage 杠杆因子: 
**LEVERAGE = 0.38 · MLEV + 0.35 · DTOA + 0.27 · BLEV**

- **MLEV**: The sum of Market Value of Equity (ME), Book Value of Preferred Equity (PE), and Long-term Debt (LD) divided by Market Value of Equity (ME)
  - **ME**: Market value of equity of the last trade day
  - **PE**: Most recent book value per share[^8] multiplied by the most recent book amount of preferred equity[^9]
  - **LD**: Most recent book value of long-term debt[^10]

- **DTOA**: Total Debt (TD) divided by Total Assets (TA)
  - **TD**: Most recent book value of total debt[^11]
  - **TA**: Most recent book value of total assets[^12]

- **BLEV**: The sum of Book Value of Common Equity (BE), Book Value of Preferred Equity (PE), and Long-term Debt (LD) divided by Book Value of Common Equity (BE)
  - **BE**: Market value of equity divided by close price, then multiplied by the most recent book value per share

[^8]: S_FA_BPS in ASHAREFINANCIALINDICATOR  
[^9]: S_SHARE_NTRD_PRFSHARE in ASHARECAPITALIZATION  
[^10]: LT_BORROW in ASHAREBALANCESHEET  
[^11]: TOT_LIAB in ASHAREBALANCESHEET  
[^12]: TOT_ASSETS in ASHAREBALANCESHEET

#### Liquidity 流动性因子：
- **LIQUIDITY =  0.35 · STOM + 0.35 · STOQ + 0.30 · STOA**
  - **STOM**: Trailing 1 month turnover
  - **STOQ**: Trailing average of 3 month turnover
  - **STOA**: Trailing average of 12 month turnover

#### Non-linear Size 非线性市值因子
- **NLSIZE: The natural log of market cap is (1) cubed; (2)orthogonalized to Size factor; (3) winsorized and standardized**

## barra_factor_earnings.py 
- Update style factor earnings to todays's date
- 注意：读取因子暴露时使用的是更新后的因子暴露，路径应与barra_cne5_factor.py中的保持一致。

The implementation includes functions to:

- Define the objective function(weighted OLS regression) and constraints(market cap weighted industry factor earnings sum to 0).
- Perform cross-sectional weighted OLS regression.
- Calculate and analyze factor returns and their significance.
- Handle residual analysis for model validation.
- Panel regression for model validation and prediction. 

## barra_risk_analysis.py
- Use Barra factor earnings to break down risk/return sources of portfolio
- Assess profitability by analyzing significance and stability of portfolio alpha

## DecileAnalysis
- Perform empirical decile analysis on each Barra style factors
- Create decile returns plot and long-short returns plot to generally understand factor returns
- Long-short returns are kept postive by longing the decile with highest return and short that with lowest return

## long_short_factor_earnings.py
- Fix the factor exposure at 1 and calculate long-short factor earnings for each style factor
- Long decile 5 w/ the largest factor exposure, and short decile 1 w/ lowest, may lead to negative returns to align with the regression earnings for easier comparing

## Contact Info
Discussions are welcomed! Please find me via the following contact info if there is any questions or comments:
- LinkedIn: https://www.linkedin.com/in/xinyue6688
- Wechat: hichloe00
