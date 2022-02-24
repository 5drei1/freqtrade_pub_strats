from datetime import datetime
from datetime import timedelta
from functools import reduce

import freqtrade.vendor.qtpylib.indicators as qtpylib
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from pandas import DataFrame
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter

buy_params ={
     "s1_ema_xs" : 3,
     "s1_ema_sm" : 5,
     "s1_ema_md" : 10,
     "s1_ema_xl" : 50,
     "s1_ema_xxl" : 240,
     "s2_ema_input" : 50,
     "s2_ema_offset_input" : -1,
     "s2_bb_sma_length" : 49,
     "s2_bb_std_dev_length" : 64,
     "s2_bb_lower_offset" : 3,
     "s2_fib_sma_len" : 50, 
     "s2_fib_atr_len" : 14,
     "s2_fib_lower_value" : 4.236,
     "s3_ema_long" : 50,
     "s3_ema_short" : 20,
     "s3_ma_fast" : 10,
     "s3_ma_slow" : 20
 }

def to_minutes(**timdelta_kwargs):
    return int(timedelta(**timdelta_kwargs).total_seconds() / 60)


class Apollo11(IStrategy):
    # Strategy created by Shane Jones https://twitter.com/shanejones
    #
    # Assited by a number of contributors https://github.com/shanejones/goddard/graphs/contributors
    #
    # Original repo hosted at https://github.com/shanejones/goddard
    timeframe = "15m"

    # Stoploss
    stoploss = -0.16
    startup_candle_count: int = 480
    trailing_stop = False
    use_custom_stoploss = True
    use_sell_signal = False

    # signal controls
    buy_signal_1 = True
    buy_signal_2 = True
    buy_signal_3 = True

    # ROI table:
    minimal_roi = {
        "0": 10,  # This is 10000%, which basically disables ROI
    }

    # Indicator values:


    # Signal 1
#    s1_ema_xs = 3
#    s1_ema_sm = 5
#    s1_ema_md = 10
#    s1_ema_xl = 50
#    s1_ema_xxl = 240

    # Signal 2
#    s2_ema_input = 50
#    s2_ema_offset_input = -1

#    s2_bb_sma_length = 49
#    s2_bb_std_dev_length = 64
#    s2_bb_lower_offset = 3

#    s2_fib_sma_len = 50
#    s2_fib_atr_len = 14
#
    s2_fib_lower_value = 4.236#

#    s3_ema_long = 50
#    s3_ema_short = 20
#    s3_ma_fast = 10
#    s3_ma_slow = 20


    s1_ema_xs = IntParameter(1, 4, default=buy_params['s1_ema_xs'], space='buy', optimize=True)
    s1_ema_sm = IntParameter(5, 9, default=buy_params['s1_ema_sm'], space='buy', optimize=True)
    s1_ema_md = IntParameter(10, 49, default=buy_params['s1_ema_md'], space='buy', optimize=True)
    s1_ema_xl = IntParameter(50, 199, default=buy_params['s1_ema_xl'], space='buy', optimize=True)
    s1_ema_xxl = IntParameter(200, 340, default=buy_params['s1_ema_xxl'], space='buy', optimize=True)

    # Signal 2
    s2_ema_input = IntParameter(10, 49, default=buy_params['s2_ema_input'], space='buy', optimize=True)
    s2_ema_offset_input = DecimalParameter(-2.0, 0.0, default=buy_params['s2_ema_offset_input'], space='buy', optimize=True)
    s2_bb_sma_length = IntParameter(1, 100, default=buy_params['s2_bb_sma_length'], space='buy', optimize=True)
    s2_bb_std_dev_length = IntParameter(1, 100, default=buy_params['s2_bb_std_dev_length'], space='buy', optimize=True)
    s2_bb_lower_offset = IntParameter(1, 6, default=buy_params['s2_bb_lower_offset'], space='buy', optimize=True)

    s2_fib_sma_len = IntParameter(1, 100, default=buy_params['s2_fib_sma_len'], space='buy', optimize=True)
    s2_fib_atr_len = IntParameter(1, 100, default=buy_params['s2_fib_atr_len'], space='buy', optimize=True)
#    s2_fib_lower_value = DecimalParameter(1.0, 5.0, default=buy_params['s2_fib_lower_value'], space='buy', optimize=True)
    s3_ema_long = IntParameter(50, 100, default=buy_params['s3_ema_long'], space='buy', optimize=True)
    s3_ema_short = IntParameter(1, 39, default=buy_params['s3_ema_short'], space='buy', optimize=True)
    s3_ma_fast = IntParameter(4, 14, default=buy_params['s3_ma_fast'], space='buy', optimize=True)
    s3_ma_slow = IntParameter(15, 25, default=buy_params['s3_ma_slow'], space='buy', optimize=True)

    @property
    def protections(self):
        return [
            {
                # Don't enter a trade right after selling a trade.
                "method": "CooldownPeriod",
                "stop_duration": to_minutes(minutes=0),
            },
            {
                # Stop trading if max-drawdown is reached.
                "method": "MaxDrawdown",
                "lookback_period": to_minutes(hours=12),
                "trade_limit": 20,  # Considering all pairs that have a minimum of 20 trades
                "stop_duration": to_minutes(hours=1),
                "max_allowed_drawdown": 0.2,  # If max-drawdown is > 20% this will activate
            },
            {
                # Stop trading if a certain amount of stoploss occurred within a certain time window.
                "method": "StoplossGuard",
                "lookback_period": to_minutes(hours=6),
                "trade_limit": 4,  # Considering all pairs that have a minimum of 4 trades
                "stop_duration": to_minutes(minutes=30),
                "only_per_pair": False,  # Looks at all pairs
            },
            {
                # Lock pairs with low profits
                "method": "LowProfitPairs",
                "lookback_period": to_minutes(hours=1, minutes=30),
                "trade_limit": 2,  # Considering all pairs that have a minimum of 2 trades
                "stop_duration": to_minutes(hours=15),
                "required_profit": 0.02,  # If profit < 2% this will activate for a pair
            },
            {
                # Lock pairs with low profits
                "method": "LowProfitPairs",
                "lookback_period": to_minutes(hours=6),
                "trade_limit": 4,  # Considering all pairs that have a minimum of 4 trades
                "stop_duration": to_minutes(minutes=30),
                "required_profit": 0.01,  # If profit < 1% this will activate for a pair
            },
        ]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Adding EMA's into the dataframe
         dataframe["s1_ema_xs"] = ta.EMA(dataframe, timeperiod=buy_params['s1_ema_xs'])
         dataframe["s1_ema_sm"] = ta.EMA(dataframe, timeperiod=buy_params['s1_ema_sm'])
         dataframe["s1_ema_md"] = ta.EMA(dataframe, timeperiod=buy_params['s1_ema_md'])
         dataframe["s1_ema_xl"] = ta.EMA(dataframe, timeperiod=buy_params['s1_ema_xl'])
         dataframe["s1_ema_xxl"] = ta.EMA(dataframe, timeperiod=buy_params['s1_ema_xxl'])

         s2_ema_value = ta.EMA(dataframe, timeperiod=buy_params['s2_ema_input'])
         s2_ema_xxl_value = ta.EMA(dataframe, timeperiod=200)
         dataframe["s2_ema"] = s2_ema_value - s2_ema_value * buy_params['s2_ema_input']

         dataframe["s2_ema_xxl_off"] = s2_ema_xxl_value - s2_ema_xxl_value * self.s2_fib_lower_value
         dataframe["s2_ema_xxl"] = ta.EMA(dataframe, timeperiod=200)

         s2_bb_sma_value = ta.SMA(dataframe, timeperiod=buy_params['s2_bb_sma_length'])
         s2_bb_std_dev_value = ta.STDDEV(dataframe, buy_params['s2_bb_std_dev_length'])
         dataframe["s2_bb_std_dev_value"] = s2_bb_std_dev_value
         dataframe["s2_bb_lower_band"] = s2_bb_sma_value - (s2_bb_std_dev_value * buy_params['s2_bb_lower_offset'])

         s2_fib_atr_value = ta.ATR(dataframe, timeframe=buy_params['s2_fib_atr_len'])
         s2_fib_sma_value = ta.SMA(dataframe, timeperiod=buy_params['s2_fib_sma_len'])

         dataframe["s2_fib_lower_band"] = s2_fib_sma_value - s2_fib_atr_value * self.s2_fib_lower_value

         s3_bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=3)
         dataframe["s3_bb_lowerband"] = s3_bollinger["lower"]

         dataframe["s3_ema_long"] = ta.EMA(dataframe, timeperiod=buy_params['s3_ema_long'])
         dataframe["s3_ema_short"] = ta.EMA(dataframe, timeperiod=buy_params['s3_ema_short'])
         dataframe["s3_fast_ma"] = ta.EMA(dataframe["volume"] * dataframe["close"], buy_params['s3_ma_fast']) / ta.EMA(dataframe["volume"], buy_params['s3_ma_fast'])
         dataframe["s3_slow_ma"] = ta.EMA(dataframe["volume"] * dataframe["close"], buy_params['s3_ma_slow']) / ta.EMA(dataframe["volume"], buy_params['s3_ma_slow'])

        # Volume weighted MACD
         dataframe["fastMA"] = ta.EMA(dataframe["volume"] * dataframe["close"], 12) / ta.EMA(dataframe["volume"], 12)
         dataframe["slowMA"] = ta.EMA(dataframe["volume"] * dataframe["close"], 26) / ta.EMA(dataframe["volume"], 26)
         dataframe["vwmacd"] = dataframe["fastMA"] - dataframe["slowMA"]
         dataframe["signal"] = ta.EMA(dataframe["vwmacd"], 9)
         dataframe["hist"] = dataframe["vwmacd"] - dataframe["signal"]

         return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # basic buy methods to keep the strategy simple

        if self.buy_signal_1:
            conditions = [
                dataframe["vwmacd"] < dataframe["signal"],
                dataframe["low"] < dataframe["s1_ema_xxl"],
                dataframe["close"] > dataframe["s1_ema_xxl"],
                qtpylib.crossed_above(dataframe["s1_ema_sm"], dataframe["s1_ema_md"]),
                dataframe["s1_ema_xs"] < dataframe["s1_ema_xl"],
                dataframe["volume"] > 0,
            ]
            dataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_1")

        if self.buy_signal_2:
            conditions = [
                qtpylib.crossed_above(dataframe["s2_fib_lower_band"], dataframe["s2_bb_lower_band"]),
                dataframe["close"] < dataframe["s2_ema"],
                dataframe["volume"] > 0,
            ]
            dataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_2")

        if self.buy_signal_3:
            conditions = [
                dataframe["low"] < dataframe["s3_bb_lowerband"],
                dataframe["high"] > dataframe["s3_slow_ma"],
                dataframe["high"] < dataframe["s3_ema_long"],
                dataframe["volume"] > 0,
            ]
            dataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_3")

        if not all([self.buy_signal_1, self.buy_signal_2, self.buy_signal_3]):
            dataframe.loc[(), "buy"] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # This is essentailly ignored as we're using strict ROI / Stoploss / TTP sale scenarios
        dataframe.loc[(), "sell"] = 0
        return dataframe

    def custom_stoploss(
        self, pair: str, trade: Trade, current_time: datetime, current_rate: float, current_profit: float, **kwargs
    ) -> float:

        if current_profit > 0.2:
            return 0.04
        if current_profit > 0.1:
            return 0.03
        if current_profit > 0.06:
            return 0.02
        if current_profit > 0.03:
            return 0.01

        # Let's try to minimize the loss
        if current_profit <= -0.10:
            if trade.open_date_utc + timedelta(hours=60) < current_time:
                # After 60H since buy
                return current_profit / 1.75

        if current_profit <= -0.08:
            if trade.open_date_utc + timedelta(hours=120) < current_time:
                # After 120H since buy
                return current_profit / 1.70

        return -1
