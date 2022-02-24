# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from functools import reduce
from pandas import DataFrame, Series
# --------------------------------

import talib.abstract as ta
import numpy as np
import freqtrade.vendor.qtpylib.indicators as qtpylib
import datetime
from technical.util import resample_to_interval, resampled_merge
from datetime import datetime, timedelta
from freqtrade.persistence import Trade
from freqtrade.strategy import stoploss_from_open, merge_informative_pair, DecimalParameter, IntParameter, CategoricalParameter
from technical.indicators import VIDYA, vwma, zema

import pandas as pd
#pd.options.mode.chained_assignment = None

import math


###########################################################################################################
##                NostalgiaForInfinityV7 by iterativ                                                     ##
##                                                                                                       ##
##    Strategy for Freqtrade https://github.com/freqtrade/freqtrade                                      ##
##                                                                                                       ##
###########################################################################################################
##               GENERAL RECOMMENDATIONS                                                                 ##
##                                                                                                       ##
##   For optimal performance, suggested to use between 4 and 6 open trades, with unlimited stake.        ##
##   A pairlist with 40 to 80 pairs. Volume pairlist works well.                                         ##
##   Prefer stable coin (USDT, BUSDT etc) pairs, instead of BTC or ETH pairs.                            ##
##   Highly recommended to blacklist leveraged tokens (*BULL, *BEAR, *UP, *DOWN etc).                    ##
##   Ensure that you don't override any variables in you config.json. Especially                         ##
##   the timeframe (must be 5m).                                                                         ##
##     use_sell_signal must set to true (or not set at all).                                             ##
##     sell_profit_only must set to false (or not set at all).                                           ##
##     ignore_roi_if_buy_signal must set to true (or not set at all).                                    ##
##                                                                                                       ##
###########################################################################################################
##               DONATIONS                                                                               ##
##                                                                                                       ##
##   Absolutely not required. However, will be accepted as a token of appreciation.                      ##
##                                                                                                       ##
##   BTC: bc1qvflsvddkmxh7eqhc4jyu5z5k6xcw3ay8jl49sk                                                     ##
##   ETH (ERC20): 0x83D3cFb8001BDC5d2211cBeBB8cB3461E5f7Ec91                                             ##
##   BEP20/BSC (ETH, BNB, ...): 0x86A0B21a20b39d16424B7c8003E4A7e12d78ABEe                               ##
##                                                                                                       ##
###########################################################################################################
pmaxtype = 1
pmaxl = 10
pmaxmulti = 3
pmaxperiod = 10
mavalue = 'MA_' + str(pmaxtype)+ '_' + str(pmaxl)

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema 

def zlema(dataframe, timeperiod):
    lag =  int(math.floor((timeperiod - 1) / 2) )
    if isinstance(dataframe, pd.Series):
        ema_data = dataframe  + (dataframe  - dataframe.shift(lag))
    else:
        ema_data = dataframe['close']  + (dataframe['close']  - dataframe['close'] .shift(lag))
    return ta.EMA(ema_data, timeperiod = timeperiod)

def zlhull(dataframe, timeperiod):
    lag =  int(math.floor((timeperiod - 1) / 2) )
    if isinstance(dataframe, pd.Series):
        wma_data = dataframe + (dataframe  - dataframe.shift(lag))
    else:
        wma_data = dataframe['close'] + (dataframe['close']  - dataframe['close'] .shift(lag))

    return  ta.WMA(
                    2 * ta.WMA(wma_data, int(math.floor(timeperiod/2))) - ta.WMA(wma_data, timeperiod), int(round(np.sqrt(timeperiod)))
                    )

def hull(dataframe, timeperiod):
    if isinstance(dataframe, pd.Series):
        return  ta.WMA(
                    2 * ta.WMA(dataframe, int(math.floor(timeperiod/2))) - ta.WMA(dataframe, timeperiod), int(round(np.sqrt(timeperiod)))
                    )

    else:
        return  ta.WMA(
                    2 * ta.WMA(dataframe['close'], int(math.floor(timeperiod/2))) - ta.WMA(dataframe['close'], timeperiod), int(round(np.sqrt(timeperiod)))
                    )


ma_types = {
    'SMA': ta.SMA,
#    'EMA': ta.EMA,
    'zlema': zlema,
#    'zlhull':zlhull ,
#    'WMA' : ta.WMA,
    'hull' : hull,
#    'TEMA': ta.TEMA,
#    'DEMA': ta.DEMA,
#    'TRIMA':ta.TRIMA,
#    'KAMA':ta.KAMA,
    'T3':ta.T3 ,


}

class Comb_ho4_7_H9(IStrategy):
    INTERFACE_VERSION = 2

    # # ROI table:
    minimal_roi = {
        "0": 10,
    }

    stoploss = -0.99

    # Trailing stoploss (not used)
    trailing_stop = False
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03

    use_custom_stoploss = False

    # Optimal timeframe for the strategy.
    timeframe = '5m'
    inf_1h = '1h'

    custom_info = {}

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 400

    # Optional order type mapping.
    order_types = {
        'buy': 'limit',
        'sell': 'limit',
        'trailing_stop_loss': 'limit',
        'stoploss': 'limit',
        'stoploss_on_exchange': False
    }

    #############################################################

    # Buy hyperspace params:
    buy_params = {
        "base_nb_candles_bearish_buy": 52,
        "base_nb_candles_bullish_buy": 77,
        "ewo_high": 10.996,
        "ewo_low": -14.081,
        "low_offset_bearish": 0.96,
        "low_offset_bullish": 0.976,
        "pmaxmultiplier": 33,
        "rsi_buy_bearish": 39,
        "rsi_buy_bullish": 56,
        "buy_01_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_01_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_01_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_01_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_01_protection__ema_fast": False,  # value loaded from strategy
        "buy_01_protection__ema_fast_len": "26",  # value loaded from strategy
        "buy_01_protection__ema_slow": True,  # value loaded from strategy
        "buy_01_protection__ema_slow_len": "100",  # value loaded from strategy
        "buy_01_protection__safe_dips": True,  # value loaded from strategy
        "buy_01_protection__safe_dips_type": "80",  # value loaded from strategy
        "buy_01_protection__safe_pump": True,  # value loaded from strategy
        "buy_01_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_01_protection__safe_pump_type": "70",  # value loaded from strategy
        "buy_01_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_01_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_01_protection__sma200_rising": True,  # value loaded from strategy
        "buy_01_protection__sma200_rising_val": "28",  # value loaded from strategy
        "buy_02_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_02_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_02_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_02_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_02_protection__ema_fast": False,  # value loaded from strategy
        "buy_02_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_02_protection__ema_slow": True,  # value loaded from strategy
        "buy_02_protection__ema_slow_len": "20",  # value loaded from strategy
        "buy_02_protection__safe_dips": True,  # value loaded from strategy
        "buy_02_protection__safe_dips_type": "50",  # value loaded from strategy
        "buy_02_protection__safe_pump": False,  # value loaded from strategy
        "buy_02_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_02_protection__safe_pump_type": "10",  # value loaded from strategy
        "buy_02_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_02_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_02_protection__sma200_rising": False,  # value loaded from strategy
        "buy_02_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_03_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_03_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_03_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_03_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_03_protection__ema_fast": True,  # value loaded from strategy
        "buy_03_protection__ema_fast_len": "100",  # value loaded from strategy
        "buy_03_protection__ema_slow": True,  # value loaded from strategy
        "buy_03_protection__ema_slow_len": "100",  # value loaded from strategy
        "buy_03_protection__safe_dips": False,  # value loaded from strategy
        "buy_03_protection__safe_dips_type": "10",  # value loaded from strategy
        "buy_03_protection__safe_pump": True,  # value loaded from strategy
        "buy_03_protection__safe_pump_period": "36",  # value loaded from strategy
        "buy_03_protection__safe_pump_type": "100",  # value loaded from strategy
        "buy_03_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_03_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_03_protection__sma200_rising": False,  # value loaded from strategy
        "buy_03_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_04_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_04_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_04_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_04_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_04_protection__ema_fast": False,  # value loaded from strategy
        "buy_04_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_04_protection__ema_slow": True,  # value loaded from strategy
        "buy_04_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_04_protection__safe_dips": True,  # value loaded from strategy
        "buy_04_protection__safe_dips_type": "40",  # value loaded from strategy
        "buy_04_protection__safe_pump": True,  # value loaded from strategy
        "buy_04_protection__safe_pump_period": "48",  # value loaded from strategy
        "buy_04_protection__safe_pump_type": "110",  # value loaded from strategy
        "buy_04_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_04_protection__sma200_1h_rising_val": "20",  # value loaded from strategy
        "buy_04_protection__sma200_rising": True,  # value loaded from strategy
        "buy_04_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_05_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_05_protection__close_above_ema_fast_len": "100",  # value loaded from strategy
        "buy_05_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_05_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_05_protection__ema_fast": True,  # value loaded from strategy
        "buy_05_protection__ema_fast_len": "100",  # value loaded from strategy
        "buy_05_protection__ema_slow": False,  # value loaded from strategy
        "buy_05_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_05_protection__safe_dips": True,  # value loaded from strategy
        "buy_05_protection__safe_dips_type": "100",  # value loaded from strategy
        "buy_05_protection__safe_pump": True,  # value loaded from strategy
        "buy_05_protection__safe_pump_period": "36",  # value loaded from strategy
        "buy_05_protection__safe_pump_type": "20",  # value loaded from strategy
        "buy_05_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_05_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_05_protection__sma200_rising": False,  # value loaded from strategy
        "buy_05_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_06_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_06_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_06_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_06_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_06_protection__ema_fast": False,  # value loaded from strategy
        "buy_06_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_06_protection__ema_slow": True,  # value loaded from strategy
        "buy_06_protection__ema_slow_len": "100",  # value loaded from strategy
        "buy_06_protection__safe_dips": True,  # value loaded from strategy
        "buy_06_protection__safe_dips_type": "50",  # value loaded from strategy
        "buy_06_protection__safe_pump": True,  # value loaded from strategy
        "buy_06_protection__safe_pump_period": "36",  # value loaded from strategy
        "buy_06_protection__safe_pump_type": "20",  # value loaded from strategy
        "buy_06_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_06_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_06_protection__sma200_rising": False,  # value loaded from strategy
        "buy_06_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_07_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_07_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_07_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_07_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_07_protection__ema_fast": True,  # value loaded from strategy
        "buy_07_protection__ema_fast_len": "100",  # value loaded from strategy
        "buy_07_protection__ema_slow": True,  # value loaded from strategy
        "buy_07_protection__ema_slow_len": "12",  # value loaded from strategy
        "buy_07_protection__safe_dips": False,  # value loaded from strategy
        "buy_07_protection__safe_dips_type": "50",  # value loaded from strategy
        "buy_07_protection__safe_pump": False,  # value loaded from strategy
        "buy_07_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_07_protection__safe_pump_type": "50",  # value loaded from strategy
        "buy_07_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_07_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_07_protection__sma200_rising": False,  # value loaded from strategy
        "buy_07_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_08_protection__close_above_ema_fast": True,  # value loaded from strategy
        "buy_08_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_08_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_08_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_08_protection__ema_fast": False,  # value loaded from strategy
        "buy_08_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_08_protection__ema_slow": True,  # value loaded from strategy
        "buy_08_protection__ema_slow_len": "12",  # value loaded from strategy
        "buy_08_protection__safe_dips": False,  # value loaded from strategy
        "buy_08_protection__safe_dips_type": "100",  # value loaded from strategy
        "buy_08_protection__safe_pump": True,  # value loaded from strategy
        "buy_08_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_08_protection__safe_pump_type": "120",  # value loaded from strategy
        "buy_08_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_08_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_08_protection__sma200_rising": False,  # value loaded from strategy
        "buy_08_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_09_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_09_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_09_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_09_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_09_protection__ema_fast": True,  # value loaded from strategy
        "buy_09_protection__ema_fast_len": "100",  # value loaded from strategy
        "buy_09_protection__ema_slow": False,  # value loaded from strategy
        "buy_09_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_09_protection__safe_dips": False,  # value loaded from strategy
        "buy_09_protection__safe_dips_type": "10",  # value loaded from strategy
        "buy_09_protection__safe_pump": False,  # value loaded from strategy
        "buy_09_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_09_protection__safe_pump_type": "50",  # value loaded from strategy
        "buy_09_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_09_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_09_protection__sma200_rising": False,  # value loaded from strategy
        "buy_09_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_10_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_10_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_10_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_10_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_10_protection__ema_fast": False,  # value loaded from strategy
        "buy_10_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_10_protection__ema_slow": True,  # value loaded from strategy
        "buy_10_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_10_protection__safe_dips": True,  # value loaded from strategy
        "buy_10_protection__safe_dips_type": "100",  # value loaded from strategy
        "buy_10_protection__safe_pump": False,  # value loaded from strategy
        "buy_10_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_10_protection__safe_pump_type": "50",  # value loaded from strategy
        "buy_10_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_10_protection__sma200_1h_rising_val": "24",  # value loaded from strategy
        "buy_10_protection__sma200_rising": False,  # value loaded from strategy
        "buy_10_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_11_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_11_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_11_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_11_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_11_protection__ema_fast": False,  # value loaded from strategy
        "buy_11_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_11_protection__ema_slow": False,  # value loaded from strategy
        "buy_11_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_11_protection__safe_dips": False,  # value loaded from strategy
        "buy_11_protection__safe_dips_type": "100",  # value loaded from strategy
        "buy_11_protection__safe_pump": True,  # value loaded from strategy
        "buy_11_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_11_protection__safe_pump_type": "50",  # value loaded from strategy
        "buy_11_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_11_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_11_protection__sma200_rising": False,  # value loaded from strategy
        "buy_11_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_12_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_12_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_12_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_12_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_12_protection__ema_fast": False,  # value loaded from strategy
        "buy_12_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_12_protection__ema_slow": False,  # value loaded from strategy
        "buy_12_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_12_protection__safe_dips": False,  # value loaded from strategy
        "buy_12_protection__safe_dips_type": "100",  # value loaded from strategy
        "buy_12_protection__safe_pump": False,  # value loaded from strategy
        "buy_12_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_12_protection__safe_pump_type": "50",  # value loaded from strategy
        "buy_12_protection__sma200_1h_rising": True,  # value loaded from strategy
        "buy_12_protection__sma200_1h_rising_val": "24",  # value loaded from strategy
        "buy_12_protection__sma200_rising": False,  # value loaded from strategy
        "buy_12_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_13_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_13_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_13_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_13_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_13_protection__ema_fast": False,  # value loaded from strategy
        "buy_13_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_13_protection__ema_slow": False,  # value loaded from strategy
        "buy_13_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_13_protection__safe_dips": True,  # value loaded from strategy
        "buy_13_protection__safe_dips_type": "10",  # value loaded from strategy
        "buy_13_protection__safe_pump": False,  # value loaded from strategy
        "buy_13_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_13_protection__safe_pump_type": "50",  # value loaded from strategy
        "buy_13_protection__sma200_1h_rising": True,  # value loaded from strategy
        "buy_13_protection__sma200_1h_rising_val": "24",  # value loaded from strategy
        "buy_13_protection__sma200_rising": False,  # value loaded from strategy
        "buy_13_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_14_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_14_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_14_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_14_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_14_protection__ema_fast": False,  # value loaded from strategy
        "buy_14_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_14_protection__ema_slow": False,  # value loaded from strategy
        "buy_14_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_14_protection__safe_dips": True,  # value loaded from strategy
        "buy_14_protection__safe_dips_type": "10",  # value loaded from strategy
        "buy_14_protection__safe_pump": True,  # value loaded from strategy
        "buy_14_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_14_protection__safe_pump_type": "70",  # value loaded from strategy
        "buy_14_protection__sma200_1h_rising": True,  # value loaded from strategy
        "buy_14_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_14_protection__sma200_rising": True,  # value loaded from strategy
        "buy_14_protection__sma200_rising_val": "30",  # value loaded from strategy
        "buy_15_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_15_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_15_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_15_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_15_protection__ema_fast": False,  # value loaded from strategy
        "buy_15_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_15_protection__ema_slow": True,  # value loaded from strategy
        "buy_15_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_15_protection__safe_dips": True,  # value loaded from strategy
        "buy_15_protection__safe_dips_type": "50",  # value loaded from strategy
        "buy_15_protection__safe_pump": True,  # value loaded from strategy
        "buy_15_protection__safe_pump_period": "36",  # value loaded from strategy
        "buy_15_protection__safe_pump_type": "10",  # value loaded from strategy
        "buy_15_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_15_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_15_protection__sma200_rising": False,  # value loaded from strategy
        "buy_15_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_16_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_16_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_16_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_16_protection__close_above_ema_slow_len": "50",  # value loaded from strategy
        "buy_16_protection__ema_fast": False,  # value loaded from strategy
        "buy_16_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_16_protection__ema_slow": True,  # value loaded from strategy
        "buy_16_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_16_protection__safe_dips": True,  # value loaded from strategy
        "buy_16_protection__safe_dips_type": "10",  # value loaded from strategy
        "buy_16_protection__safe_pump": True,  # value loaded from strategy
        "buy_16_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_16_protection__safe_pump_type": "10",  # value loaded from strategy
        "buy_16_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_16_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_16_protection__sma200_rising": False,  # value loaded from strategy
        "buy_16_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_17_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_17_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_17_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_17_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_17_protection__ema_fast": False,  # value loaded from strategy
        "buy_17_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_17_protection__ema_slow": False,  # value loaded from strategy
        "buy_17_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_17_protection__safe_dips": True,  # value loaded from strategy
        "buy_17_protection__safe_dips_type": "10",  # value loaded from strategy
        "buy_17_protection__safe_pump": True,  # value loaded from strategy
        "buy_17_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_17_protection__safe_pump_type": "120",  # value loaded from strategy
        "buy_17_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_17_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_17_protection__sma200_rising": False,  # value loaded from strategy
        "buy_17_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_18_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_18_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_18_protection__close_above_ema_slow": True,  # value loaded from strategy
        "buy_18_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_18_protection__ema_fast": True,  # value loaded from strategy
        "buy_18_protection__ema_fast_len": "100",  # value loaded from strategy
        "buy_18_protection__ema_slow": True,  # value loaded from strategy
        "buy_18_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_18_protection__safe_dips": True,  # value loaded from strategy
        "buy_18_protection__safe_dips_type": "50",  # value loaded from strategy
        "buy_18_protection__safe_pump": True,  # value loaded from strategy
        "buy_18_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_18_protection__safe_pump_type": "60",  # value loaded from strategy
        "buy_18_protection__sma200_1h_rising": True,  # value loaded from strategy
        "buy_18_protection__sma200_1h_rising_val": "72",  # value loaded from strategy
        "buy_18_protection__sma200_rising": True,  # value loaded from strategy
        "buy_18_protection__sma200_rising_val": "44",  # value loaded from strategy
        "buy_19_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_19_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_19_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_19_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_19_protection__ema_fast": False,  # value loaded from strategy
        "buy_19_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_19_protection__ema_slow": True,  # value loaded from strategy
        "buy_19_protection__ema_slow_len": "100",  # value loaded from strategy
        "buy_19_protection__safe_dips": True,  # value loaded from strategy
        "buy_19_protection__safe_dips_type": "50",  # value loaded from strategy
        "buy_19_protection__safe_pump": True,  # value loaded from strategy
        "buy_19_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_19_protection__safe_pump_type": "50",  # value loaded from strategy
        "buy_19_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_19_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_19_protection__sma200_rising": True,  # value loaded from strategy
        "buy_19_protection__sma200_rising_val": "36",  # value loaded from strategy
        "buy_20_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_20_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_20_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_20_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_20_protection__ema_fast": False,  # value loaded from strategy
        "buy_20_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_20_protection__ema_slow": True,  # value loaded from strategy
        "buy_20_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_20_protection__safe_dips": False,  # value loaded from strategy
        "buy_20_protection__safe_dips_type": "10",  # value loaded from strategy
        "buy_20_protection__safe_pump": False,  # value loaded from strategy
        "buy_20_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_20_protection__safe_pump_type": "50",  # value loaded from strategy
        "buy_20_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_20_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_20_protection__sma200_rising": False,  # value loaded from strategy
        "buy_20_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_21_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_21_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_21_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_21_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_21_protection__ema_fast": False,  # value loaded from strategy
        "buy_21_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_21_protection__ema_slow": True,  # value loaded from strategy
        "buy_21_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_21_protection__safe_dips": True,  # value loaded from strategy
        "buy_21_protection__safe_dips_type": "90",  # value loaded from strategy
        "buy_21_protection__safe_pump": False,  # value loaded from strategy
        "buy_21_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_21_protection__safe_pump_type": "50",  # value loaded from strategy
        "buy_21_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_21_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_21_protection__sma200_rising": False,  # value loaded from strategy
        "buy_21_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_22_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_22_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_22_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_22_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_22_protection__ema_fast": False,  # value loaded from strategy
        "buy_22_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_22_protection__ema_slow": False,  # value loaded from strategy
        "buy_22_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_22_protection__safe_dips": False,  # value loaded from strategy
        "buy_22_protection__safe_dips_type": "10",  # value loaded from strategy
        "buy_22_protection__safe_pump": False,  # value loaded from strategy
        "buy_22_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_22_protection__safe_pump_type": "50",  # value loaded from strategy
        "buy_22_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_22_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_22_protection__sma200_rising": False,  # value loaded from strategy
        "buy_22_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_23_protection__close_above_ema_fast": True,  # value loaded from strategy
        "buy_23_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_23_protection__close_above_ema_slow": True,  # value loaded from strategy
        "buy_23_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_23_protection__ema_fast": False,  # value loaded from strategy
        "buy_23_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_23_protection__ema_slow": False,  # value loaded from strategy
        "buy_23_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_23_protection__safe_dips": True,  # value loaded from strategy
        "buy_23_protection__safe_dips_type": "100",  # value loaded from strategy
        "buy_23_protection__safe_pump": False,  # value loaded from strategy
        "buy_23_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_23_protection__safe_pump_type": "50",  # value loaded from strategy
        "buy_23_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_23_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_23_protection__sma200_rising": False,  # value loaded from strategy
        "buy_23_protection__sma200_rising_val": "50",  # value loaded from strategy
        "buy_24_protection__close_above_ema_fast": False,  # value loaded from strategy
        "buy_24_protection__close_above_ema_fast_len": "200",  # value loaded from strategy
        "buy_24_protection__close_above_ema_slow": False,  # value loaded from strategy
        "buy_24_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_24_protection__ema_fast": False,  # value loaded from strategy
        "buy_24_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_24_protection__ema_slow": False,  # value loaded from strategy
        "buy_24_protection__ema_slow_len": "50",  # value loaded from strategy
        "buy_24_protection__safe_dips": True,  # value loaded from strategy
        "buy_24_protection__safe_dips_type": "10",  # value loaded from strategy
        "buy_24_protection__safe_pump": False,  # value loaded from strategy
        "buy_24_protection__safe_pump_period": "24",  # value loaded from strategy
        "buy_24_protection__safe_pump_type": "50",  # value loaded from strategy
        "buy_24_protection__sma200_1h_rising": True,  # value loaded from strategy
        "buy_24_protection__sma200_1h_rising_val": "36",  # value loaded from strategy
        "buy_24_protection__sma200_rising": True,  # value loaded from strategy
        "buy_24_protection__sma200_rising_val": "30",  # value loaded from strategy
        "buy_24_rsi_1h_min": 66.9,  # value loaded from strategy
        "buy_24_rsi_max": 60.0,  # value loaded from strategy
        "buy_26_protection__close_above_ema_fast": True,  # value loaded from strategy
        "buy_26_protection__close_above_ema_fast_len": "50",  # value loaded from strategy
        "buy_26_protection__close_above_ema_slow": True,  # value loaded from strategy
        "buy_26_protection__close_above_ema_slow_len": "200",  # value loaded from strategy
        "buy_26_protection__ema_fast": False,  # value loaded from strategy
        "buy_26_protection__ema_fast_len": "50",  # value loaded from strategy
        "buy_26_protection__ema_slow": True,  # value loaded from strategy
        "buy_26_protection__ema_slow_len": "100",  # value loaded from strategy
        "buy_26_protection__safe_dips": True,  # value loaded from strategy
        "buy_26_protection__safe_dips_type": "10",  # value loaded from strategy
        "buy_26_protection__safe_pump": True,  # value loaded from strategy
        "buy_26_protection__safe_pump_period": "48",  # value loaded from strategy
        "buy_26_protection__safe_pump_type": "100",  # value loaded from strategy
        "buy_26_protection__sma200_1h_rising": False,  # value loaded from strategy
        "buy_26_protection__sma200_1h_rising_val": "50",  # value loaded from strategy
        "buy_26_protection__sma200_rising": False,  # value loaded from strategy
        "buy_26_protection__sma200_rising_val": "30",  # value loaded from strategy
        "buy_26_zema_low_offset": 0.93,  # value loaded from strategy
        "buy_bb20_close_bblowerband_4": 0.98,  # value loaded from strategy
        "buy_bb20_volume_4": 10.0,  # value loaded from strategy
        "buy_bb40_bbdelta_close_3": 0.059,  # value loaded from strategy
        "buy_bb40_closedelta_close_3": 0.023,  # value loaded from strategy
        "buy_bb40_tail_bbdelta_3": 0.418,  # value loaded from strategy
        "buy_bb_offset_10": 0.985,  # value loaded from strategy
        "buy_bb_offset_14": 0.988,  # value loaded from strategy
        "buy_bb_offset_18": 0.982,  # value loaded from strategy
        "buy_bb_offset_2": 0.983,  # value loaded from strategy
        "buy_bb_offset_22": 0.98,  # value loaded from strategy
        "buy_bb_offset_23": 0.987,  # value loaded from strategy
        "buy_bb_offset_5": 0.996,  # value loaded from strategy
        "buy_bb_offset_6": 0.984,  # value loaded from strategy
        "buy_bb_offset_9": 0.942,  # value loaded from strategy
        "buy_bearish_trigger": "zlema",  # value loaded from strategy
        "buy_bullish_trigger": "hull",  # value loaded from strategy
        "buy_chop_min_19": 24.1,  # value loaded from strategy
        "buy_condition_10_enable": True,  # value loaded from strategy
        "buy_condition_11_enable": True,  # value loaded from strategy
        "buy_condition_12_enable": True,  # value loaded from strategy
        "buy_condition_13_enable": True,  # value loaded from strategy
        "buy_condition_14_enable": True,  # value loaded from strategy
        "buy_condition_15_enable": True,  # value loaded from strategy
        "buy_condition_16_enable": True,  # value loaded from strategy
        "buy_condition_17_enable": True,  # value loaded from strategy
        "buy_condition_18_enable": True,  # value loaded from strategy
        "buy_condition_19_enable": True,  # value loaded from strategy
        "buy_condition_1_enable": True,  # value loaded from strategy
        "buy_condition_20_enable": True,  # value loaded from strategy
        "buy_condition_21_enable": True,  # value loaded from strategy
        "buy_condition_22_enable": True,  # value loaded from strategy
        "buy_condition_23_enable": True,  # value loaded from strategy
        "buy_condition_24_enable": True,  # value loaded from strategy
        "buy_condition_26_enable": True,  # value loaded from strategy
        "buy_condition_2_enable": True,  # value loaded from strategy
        "buy_condition_3_enable": True,  # value loaded from strategy
        "buy_condition_4_enable": True,  # value loaded from strategy
        "buy_condition_5_enable": True,  # value loaded from strategy
        "buy_condition_6_enable": True,  # value loaded from strategy
        "buy_condition_7_enable": True,  # value loaded from strategy
        "buy_condition_8_enable": True,  # value loaded from strategy
        "buy_condition_9_enable": True,  # value loaded from strategy
        "buy_dip_threshold_100_1": 0.026,  # value loaded from strategy
        "buy_dip_threshold_100_2": 0.24,  # value loaded from strategy
        "buy_dip_threshold_100_3": 0.42,  # value loaded from strategy
        "buy_dip_threshold_100_4": 0.8,  # value loaded from strategy
        "buy_dip_threshold_10_1": 0.015,  # value loaded from strategy
        "buy_dip_threshold_10_2": 0.1,  # value loaded from strategy
        "buy_dip_threshold_10_3": 0.24,  # value loaded from strategy
        "buy_dip_threshold_10_4": 0.42,  # value loaded from strategy
        "buy_dip_threshold_110_1": 0.027,  # value loaded from strategy
        "buy_dip_threshold_110_2": 0.26,  # value loaded from strategy
        "buy_dip_threshold_110_3": 0.44,  # value loaded from strategy
        "buy_dip_threshold_110_4": 0.84,  # value loaded from strategy
        "buy_dip_threshold_20_1": 0.016,  # value loaded from strategy
        "buy_dip_threshold_20_2": 0.11,  # value loaded from strategy
        "buy_dip_threshold_20_3": 0.26,  # value loaded from strategy
        "buy_dip_threshold_20_4": 0.44,  # value loaded from strategy
        "buy_dip_threshold_30_1": 0.018,  # value loaded from strategy
        "buy_dip_threshold_30_2": 0.12,  # value loaded from strategy
        "buy_dip_threshold_30_3": 0.28,  # value loaded from strategy
        "buy_dip_threshold_30_4": 0.46,  # value loaded from strategy
        "buy_dip_threshold_40_1": 0.019,  # value loaded from strategy
        "buy_dip_threshold_40_2": 0.13,  # value loaded from strategy
        "buy_dip_threshold_40_3": 0.3,  # value loaded from strategy
        "buy_dip_threshold_40_4": 0.48,  # value loaded from strategy
        "buy_dip_threshold_50_1": 0.02,  # value loaded from strategy
        "buy_dip_threshold_50_2": 0.14,  # value loaded from strategy
        "buy_dip_threshold_50_3": 0.32,  # value loaded from strategy
        "buy_dip_threshold_50_4": 0.5,  # value loaded from strategy
        "buy_dip_threshold_60_1": 0.022,  # value loaded from strategy
        "buy_dip_threshold_60_2": 0.18,  # value loaded from strategy
        "buy_dip_threshold_60_3": 0.34,  # value loaded from strategy
        "buy_dip_threshold_60_4": 0.56,  # value loaded from strategy
        "buy_dip_threshold_70_1": 0.023,  # value loaded from strategy
        "buy_dip_threshold_70_2": 0.2,  # value loaded from strategy
        "buy_dip_threshold_70_3": 0.36,  # value loaded from strategy
        "buy_dip_threshold_70_4": 0.6,  # value loaded from strategy
        "buy_dip_threshold_80_1": 0.024,  # value loaded from strategy
        "buy_dip_threshold_80_2": 0.22,  # value loaded from strategy
        "buy_dip_threshold_80_3": 0.38,  # value loaded from strategy
        "buy_dip_threshold_80_4": 0.66,  # value loaded from strategy
        "buy_dip_threshold_90_1": 0.025,  # value loaded from strategy
        "buy_dip_threshold_90_2": 0.23,  # value loaded from strategy
        "buy_dip_threshold_90_3": 0.4,  # value loaded from strategy
        "buy_dip_threshold_90_4": 0.7,  # value loaded from strategy
        "buy_ema_open_mult_14": 0.014,  # value loaded from strategy
        "buy_ema_open_mult_15": 0.018,  # value loaded from strategy
        "buy_ema_open_mult_5": 0.018,  # value loaded from strategy
        "buy_ema_open_mult_6": 0.021,  # value loaded from strategy
        "buy_ema_open_mult_7": 0.03,  # value loaded from strategy
        "buy_ema_rel_15": 0.988,  # value loaded from strategy
        "buy_ema_rel_3": 0.986,  # value loaded from strategy
        "buy_ema_rel_5": 0.944,  # value loaded from strategy
        "buy_ewo_12": 1.8,  # value loaded from strategy
        "buy_ewo_13": -11.4,  # value loaded from strategy
        "buy_ewo_16": 2.8,  # value loaded from strategy
        "buy_ewo_17": -12.8,  # value loaded from strategy
        "buy_ewo_22": 4.2,  # value loaded from strategy
        "buy_ewo_23": 7.0,  # value loaded from strategy
        "buy_ma_offset_10": 0.948,  # value loaded from strategy
        "buy_ma_offset_11": 0.934,  # value loaded from strategy
        "buy_ma_offset_12": 0.922,  # value loaded from strategy
        "buy_ma_offset_13": 0.99,  # value loaded from strategy
        "buy_ma_offset_14": 0.98,  # value loaded from strategy
        "buy_ma_offset_15": 0.954,  # value loaded from strategy
        "buy_ma_offset_16": 0.952,  # value loaded from strategy
        "buy_ma_offset_17": 0.952,  # value loaded from strategy
        "buy_ma_offset_22": 0.94,  # value loaded from strategy
        "buy_ma_offset_9": 0.922,  # value loaded from strategy
        "buy_mfi_1": 36.0,  # value loaded from strategy
        "buy_mfi_11": 36.0,  # value loaded from strategy
        "buy_mfi_2": 49.0,  # value loaded from strategy
        "buy_mfi_9": 50.0,  # value loaded from strategy
        "buy_min_inc_1": 0.022,  # value loaded from strategy
        "buy_min_inc_11": 0.01,  # value loaded from strategy
        "buy_pump_pull_threshold_100_24": 1.7,  # value loaded from strategy
        "buy_pump_pull_threshold_100_36": 1.7,  # value loaded from strategy
        "buy_pump_pull_threshold_100_48": 1.4,  # value loaded from strategy
        "buy_pump_pull_threshold_10_24": 2.2,  # value loaded from strategy
        "buy_pump_pull_threshold_10_36": 2.0,  # value loaded from strategy
        "buy_pump_pull_threshold_10_48": 2.0,  # value loaded from strategy
        "buy_pump_pull_threshold_110_24": 1.7,  # value loaded from strategy
        "buy_pump_pull_threshold_110_36": 1.7,  # value loaded from strategy
        "buy_pump_pull_threshold_110_48": 1.4,  # value loaded from strategy
        "buy_pump_pull_threshold_120_24": 1.7,  # value loaded from strategy
        "buy_pump_pull_threshold_120_36": 1.7,  # value loaded from strategy
        "buy_pump_pull_threshold_120_48": 1.4,  # value loaded from strategy
        "buy_pump_pull_threshold_20_24": 2.2,  # value loaded from strategy
        "buy_pump_pull_threshold_20_36": 2.0,  # value loaded from strategy
        "buy_pump_pull_threshold_20_48": 2.0,  # value loaded from strategy
        "buy_pump_pull_threshold_30_24": 2.2,  # value loaded from strategy
        "buy_pump_pull_threshold_30_36": 2.0,  # value loaded from strategy
        "buy_pump_pull_threshold_30_48": 2.0,  # value loaded from strategy
        "buy_pump_pull_threshold_40_24": 2.2,  # value loaded from strategy
        "buy_pump_pull_threshold_40_36": 2.0,  # value loaded from strategy
        "buy_pump_pull_threshold_40_48": 2.0,  # value loaded from strategy
        "buy_pump_pull_threshold_50_24": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_50_36": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_50_48": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_60_24": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_60_36": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_60_48": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_70_24": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_70_36": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_70_48": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_80_24": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_80_36": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_80_48": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_90_24": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_90_36": 1.75,  # value loaded from strategy
        "buy_pump_pull_threshold_90_48": 1.75,  # value loaded from strategy
        "buy_pump_threshold_100_24": 0.66,  # value loaded from strategy
        "buy_pump_threshold_100_36": 0.7,  # value loaded from strategy
        "buy_pump_threshold_100_48": 1.6,  # value loaded from strategy
        "buy_pump_threshold_10_24": 0.42,  # value loaded from strategy
        "buy_pump_threshold_10_36": 0.58,  # value loaded from strategy
        "buy_pump_threshold_10_48": 0.8,  # value loaded from strategy
        "buy_pump_threshold_110_24": 0.7,  # value loaded from strategy
        "buy_pump_threshold_110_36": 0.74,  # value loaded from strategy
        "buy_pump_threshold_110_48": 1.8,  # value loaded from strategy
        "buy_pump_threshold_120_24": 0.78,  # value loaded from strategy
        "buy_pump_threshold_120_36": 0.78,  # value loaded from strategy
        "buy_pump_threshold_120_48": 2.0,  # value loaded from strategy
        "buy_pump_threshold_20_24": 0.46,  # value loaded from strategy
        "buy_pump_threshold_20_36": 0.6,  # value loaded from strategy
        "buy_pump_threshold_20_48": 0.81,  # value loaded from strategy
        "buy_pump_threshold_30_24": 0.5,  # value loaded from strategy
        "buy_pump_threshold_30_36": 0.62,  # value loaded from strategy
        "buy_pump_threshold_30_48": 0.82,  # value loaded from strategy
        "buy_pump_threshold_40_24": 0.54,  # value loaded from strategy
        "buy_pump_threshold_40_36": 0.63,  # value loaded from strategy
        "buy_pump_threshold_40_48": 0.84,  # value loaded from strategy
        "buy_pump_threshold_50_24": 0.6,  # value loaded from strategy
        "buy_pump_threshold_50_36": 0.64,  # value loaded from strategy
        "buy_pump_threshold_50_48": 0.85,  # value loaded from strategy
        "buy_pump_threshold_60_24": 0.62,  # value loaded from strategy
        "buy_pump_threshold_60_36": 0.66,  # value loaded from strategy
        "buy_pump_threshold_60_48": 0.9,  # value loaded from strategy
        "buy_pump_threshold_70_24": 0.63,  # value loaded from strategy
        "buy_pump_threshold_70_36": 0.67,  # value loaded from strategy
        "buy_pump_threshold_70_48": 0.95,  # value loaded from strategy
        "buy_pump_threshold_80_24": 0.64,  # value loaded from strategy
        "buy_pump_threshold_80_36": 0.68,  # value loaded from strategy
        "buy_pump_threshold_80_48": 1.0,  # value loaded from strategy
        "buy_pump_threshold_90_24": 0.65,  # value loaded from strategy
        "buy_pump_threshold_90_36": 0.69,  # value loaded from strategy
        "buy_pump_threshold_90_48": 1.1,  # value loaded from strategy
        "buy_rsi_1": 36.0,  # value loaded from strategy
        "buy_rsi_11": 48.0,  # value loaded from strategy
        "buy_rsi_12": 30.0,  # value loaded from strategy
        "buy_rsi_15": 28.0,  # value loaded from strategy
        "buy_rsi_16": 31.0,  # value loaded from strategy
        "buy_rsi_18": 26.0,  # value loaded from strategy
        "buy_rsi_1h_10": 37.0,  # value loaded from strategy
        "buy_rsi_1h_20": 20.0,  # value loaded from strategy
        "buy_rsi_1h_21": 24.0,  # value loaded from strategy
        "buy_rsi_1h_23": 70.0,  # value loaded from strategy
        "buy_rsi_1h_diff_2": 39.0,  # value loaded from strategy
        "buy_rsi_1h_max_1": 84.0,  # value loaded from strategy
        "buy_rsi_1h_max_11": 84.0,  # value loaded from strategy
        "buy_rsi_1h_max_2": 84.0,  # value loaded from strategy
        "buy_rsi_1h_max_9": 88.0,  # value loaded from strategy
        "buy_rsi_1h_min_1": 30.0,  # value loaded from strategy
        "buy_rsi_1h_min_11": 55.0,  # value loaded from strategy
        "buy_rsi_1h_min_19": 50.0,  # value loaded from strategy
        "buy_rsi_1h_min_2": 32.0,  # value loaded from strategy
        "buy_rsi_1h_min_9": 30.0,  # value loaded from strategy
        "buy_rsi_20": 27.0,  # value loaded from strategy
        "buy_rsi_21": 23.0,  # value loaded from strategy
        "buy_rsi_22": 37.0,  # value loaded from strategy
        "buy_rsi_23": 30.0,  # value loaded from strategy
        "buy_rsi_7": 37.0,  # value loaded from strategy
        "buy_rsi_8": 29.0,  # value loaded from strategy
        "buy_tail_diff_8": 3.5,  # value loaded from strategy
        "buy_volume_22": 3.0,  # value loaded from strategy
        "buy_volume_8": 2.0,  # value loaded from strategy
        "fast_ewo": 50,  # value loaded from strategy
        "ma_type_index_1": 0,  # value loaded from strategy
        "ma_type_index_2": 0,  # value loaded from strategy
        "ma_type_index_3": 0,  # value loaded from strategy
        "pmaxlength": 9,  # value loaded from strategy
        "slow_ewo": 200,  # value loaded from strategy
    }

    # Sell hyperspace params:
    sell_params = {
        "atr_multiplier": 3.52,
        "atr_rolling_length": 2,
        "base_nb_candles_bearish_sell": 16,
        "base_nb_candles_bullish_sell": 12,
        "high_offset_bearish": 0.996,
        "high_offset_bullish": 1.026,
        "ma_type_index_4": 0,  # value loaded from strategy
        "ma_type_index_5": 0,  # value loaded from strategy
        "ma_type_index_6": 0,  # value loaded from strategy
        "sell_bb_relative_8": 1.1,  # value loaded from strategy
        "sell_bearish_trigger": "zlema",  # value loaded from strategy
        "sell_bullish_trigger": "hull",  # value loaded from strategy
        "sell_condition_1_enable": True,  # value loaded from strategy
        "sell_condition_2_enable": True,  # value loaded from strategy
        "sell_condition_3_enable": True,  # value loaded from strategy
        "sell_condition_4_enable": True,  # value loaded from strategy
        "sell_condition_5_enable": True,  # value loaded from strategy
        "sell_condition_6_enable": True,  # value loaded from strategy
        "sell_condition_7_enable": True,  # value loaded from strategy
        "sell_condition_8_enable": True,  # value loaded from strategy
        "sell_custom_dec_profit_max_1": 0.12,  # value loaded from strategy
        "sell_custom_dec_profit_max_2": 0.16,  # value loaded from strategy
        "sell_custom_dec_profit_min_1": 0.05,  # value loaded from strategy
        "sell_custom_dec_profit_min_2": 0.07,  # value loaded from strategy
        "sell_custom_long_duration_min_1": 900,  # value loaded from strategy
        "sell_custom_long_profit_max_1": 0.04,  # value loaded from strategy
        "sell_custom_long_profit_min_1": 0.03,  # value loaded from strategy
        "sell_custom_profit_0": 0.01,  # value loaded from strategy
        "sell_custom_profit_1": 0.02,  # value loaded from strategy
        "sell_custom_profit_10": 0.12,  # value loaded from strategy
        "sell_custom_profit_11": 0.2,  # value loaded from strategy
        "sell_custom_profit_2": 0.03,  # value loaded from strategy
        "sell_custom_profit_3": 0.04,  # value loaded from strategy
        "sell_custom_profit_4": 0.05,  # value loaded from strategy
        "sell_custom_profit_5": 0.06,  # value loaded from strategy
        "sell_custom_profit_6": 0.07,  # value loaded from strategy
        "sell_custom_profit_7": 0.08,  # value loaded from strategy
        "sell_custom_profit_8": 0.09,  # value loaded from strategy
        "sell_custom_profit_9": 0.1,  # value loaded from strategy
        "sell_custom_profit_under_rel_1": 0.024,  # value loaded from strategy
        "sell_custom_profit_under_rsi_diff_1": 4.4,  # value loaded from strategy
        "sell_custom_pump_dec_profit_max_1": 0.05,  # value loaded from strategy
        "sell_custom_pump_dec_profit_max_2": 0.06,  # value loaded from strategy
        "sell_custom_pump_dec_profit_max_3": 0.09,  # value loaded from strategy
        "sell_custom_pump_dec_profit_max_4": 0.04,  # value loaded from strategy
        "sell_custom_pump_dec_profit_min_1": 0.005,  # value loaded from strategy
        "sell_custom_pump_dec_profit_min_2": 0.04,  # value loaded from strategy
        "sell_custom_pump_dec_profit_min_3": 0.06,  # value loaded from strategy
        "sell_custom_pump_dec_profit_min_4": 0.02,  # value loaded from strategy
        "sell_custom_pump_profit_1_1": 0.01,  # value loaded from strategy
        "sell_custom_pump_profit_1_2": 0.02,  # value loaded from strategy
        "sell_custom_pump_profit_1_3": 0.04,  # value loaded from strategy
        "sell_custom_pump_profit_1_4": 0.1,  # value loaded from strategy
        "sell_custom_pump_profit_1_5": 0.2,  # value loaded from strategy
        "sell_custom_pump_profit_2_1": 0.01,  # value loaded from strategy
        "sell_custom_pump_profit_2_2": 0.02,  # value loaded from strategy
        "sell_custom_pump_profit_2_3": 0.04,  # value loaded from strategy
        "sell_custom_pump_profit_2_4": 0.1,  # value loaded from strategy
        "sell_custom_pump_profit_2_5": 0.2,  # value loaded from strategy
        "sell_custom_pump_profit_3_1": 0.01,  # value loaded from strategy
        "sell_custom_pump_profit_3_2": 0.02,  # value loaded from strategy
        "sell_custom_pump_profit_3_3": 0.04,  # value loaded from strategy
        "sell_custom_pump_profit_3_4": 0.1,  # value loaded from strategy
        "sell_custom_pump_profit_3_5": 0.2,  # value loaded from strategy
        "sell_custom_pump_rsi_1_1": 34.0,  # value loaded from strategy
        "sell_custom_pump_rsi_1_2": 40.0,  # value loaded from strategy
        "sell_custom_pump_rsi_1_3": 42.0,  # value loaded from strategy
        "sell_custom_pump_rsi_1_4": 42.0,  # value loaded from strategy
        "sell_custom_pump_rsi_1_5": 34.0,  # value loaded from strategy
        "sell_custom_pump_rsi_2_1": 34.0,  # value loaded from strategy
        "sell_custom_pump_rsi_2_2": 40.0,  # value loaded from strategy
        "sell_custom_pump_rsi_2_3": 40.0,  # value loaded from strategy
        "sell_custom_pump_rsi_2_4": 42.0,  # value loaded from strategy
        "sell_custom_pump_rsi_2_5": 34.0,  # value loaded from strategy
        "sell_custom_pump_rsi_3_1": 34.0,  # value loaded from strategy
        "sell_custom_pump_rsi_3_2": 40.0,  # value loaded from strategy
        "sell_custom_pump_rsi_3_3": 40.0,  # value loaded from strategy
        "sell_custom_pump_rsi_3_4": 42.0,  # value loaded from strategy
        "sell_custom_pump_rsi_3_5": 34.0,  # value loaded from strategy
        "sell_custom_pump_trail_down_1": 0.05,  # value loaded from strategy
        "sell_custom_pump_trail_profit_max_1": 0.07,  # value loaded from strategy
        "sell_custom_pump_trail_profit_min_1": 0.05,  # value loaded from strategy
        "sell_custom_pump_trail_rsi_max_1": 70.0,  # value loaded from strategy
        "sell_custom_pump_trail_rsi_min_1": 20.0,  # value loaded from strategy
        "sell_custom_pump_under_profit_max_1": 0.09,  # value loaded from strategy
        "sell_custom_pump_under_profit_min_1": 0.04,  # value loaded from strategy
        "sell_custom_recover_min_loss_1": 0.12,  # value loaded from strategy
        "sell_custom_recover_min_loss_2": 0.06,  # value loaded from strategy
        "sell_custom_recover_profit_1": 0.04,  # value loaded from strategy
        "sell_custom_recover_profit_max_2": 0.05,  # value loaded from strategy
        "sell_custom_recover_profit_min_2": 0.01,  # value loaded from strategy
        "sell_custom_recover_rsi_2": 46.0,  # value loaded from strategy
        "sell_custom_rsi_0": 34.0,  # value loaded from strategy
        "sell_custom_rsi_1": 35.0,  # value loaded from strategy
        "sell_custom_rsi_10": 42.0,  # value loaded from strategy
        "sell_custom_rsi_11": 34.0,  # value loaded from strategy
        "sell_custom_rsi_2": 37.0,  # value loaded from strategy
        "sell_custom_rsi_3": 42.0,  # value loaded from strategy
        "sell_custom_rsi_4": 43.0,  # value loaded from strategy
        "sell_custom_rsi_5": 45.0,  # value loaded from strategy
        "sell_custom_rsi_6": 48.0,  # value loaded from strategy
        "sell_custom_rsi_7": 54.0,  # value loaded from strategy
        "sell_custom_rsi_8": 55.0,  # value loaded from strategy
        "sell_custom_rsi_9": 54.0,  # value loaded from strategy
        "sell_custom_stoploss_long_profit_max_1": -0.04,  # value loaded from strategy
        "sell_custom_stoploss_long_profit_min_1": -0.08,  # value loaded from strategy
        "sell_custom_stoploss_long_recover_1": 0.1,  # value loaded from strategy
        "sell_custom_stoploss_long_recover_2": 0.06,  # value loaded from strategy
        "sell_custom_stoploss_long_rsi_diff_1": 4.0,  # value loaded from strategy
        "sell_custom_stoploss_long_rsi_diff_2": 40.0,  # value loaded from strategy
        "sell_custom_stoploss_pump_loss_2": -0.05,  # value loaded from strategy
        "sell_custom_stoploss_pump_loss_3": -0.12,  # value loaded from strategy
        "sell_custom_stoploss_pump_ma_offset_1": 0.94,  # value loaded from strategy
        "sell_custom_stoploss_pump_ma_offset_2": 0.92,  # value loaded from strategy
        "sell_custom_stoploss_pump_ma_offset_3": 0.88,  # value loaded from strategy
        "sell_custom_stoploss_pump_max_1": -0.01,  # value loaded from strategy
        "sell_custom_stoploss_pump_max_profit_1": 0.025,  # value loaded from strategy
        "sell_custom_stoploss_pump_max_profit_2": 0.025,  # value loaded from strategy
        "sell_custom_stoploss_pump_max_profit_3": 0.008,  # value loaded from strategy
        "sell_custom_stoploss_pump_min_1": -0.02,  # value loaded from strategy
        "sell_custom_stoploss_under_rel_1": 0.004,  # value loaded from strategy
        "sell_custom_stoploss_under_rsi_diff_1": 10.0,  # value loaded from strategy
        "sell_custom_under_profit_0": 0.01,  # value loaded from strategy
        "sell_custom_under_profit_1": 0.02,  # value loaded from strategy
        "sell_custom_under_profit_10": 0.12,  # value loaded from strategy
        "sell_custom_under_profit_11": 0.2,  # value loaded from strategy
        "sell_custom_under_profit_2": 0.03,  # value loaded from strategy
        "sell_custom_under_profit_3": 0.04,  # value loaded from strategy
        "sell_custom_under_profit_4": 0.05,  # value loaded from strategy
        "sell_custom_under_profit_5": 0.06,  # value loaded from strategy
        "sell_custom_under_profit_6": 0.07,  # value loaded from strategy
        "sell_custom_under_profit_7": 0.08,  # value loaded from strategy
        "sell_custom_under_profit_8": 0.09,  # value loaded from strategy
        "sell_custom_under_profit_9": 0.1,  # value loaded from strategy
        "sell_custom_under_rsi_0": 35.0,  # value loaded from strategy
        "sell_custom_under_rsi_1": 56.0,  # value loaded from strategy
        "sell_custom_under_rsi_10": 42.0,  # value loaded from strategy
        "sell_custom_under_rsi_11": 34.0,  # value loaded from strategy
        "sell_custom_under_rsi_2": 57.0,  # value loaded from strategy
        "sell_custom_under_rsi_3": 58.0,  # value loaded from strategy
        "sell_custom_under_rsi_4": 59.0,  # value loaded from strategy
        "sell_custom_under_rsi_5": 60.0,  # value loaded from strategy
        "sell_custom_under_rsi_6": 56.0,  # value loaded from strategy
        "sell_custom_under_rsi_7": 54.0,  # value loaded from strategy
        "sell_custom_under_rsi_8": 55.0,  # value loaded from strategy
        "sell_custom_under_rsi_9": 54.0,  # value loaded from strategy
        "sell_dual_rsi_rsi_1h_4": 79.6,  # value loaded from strategy
        "sell_dual_rsi_rsi_4": 73.4,  # value loaded from strategy
        "sell_ema_relative_5": 0.024,  # value loaded from strategy
        "sell_pump_threshold_24_1": 0.68,  # value loaded from strategy
        "sell_pump_threshold_24_2": 0.62,  # value loaded from strategy
        "sell_pump_threshold_24_3": 0.88,  # value loaded from strategy
        "sell_pump_threshold_36_1": 0.72,  # value loaded from strategy
        "sell_pump_threshold_36_2": 4.0,  # value loaded from strategy
        "sell_pump_threshold_36_3": 1.0,  # value loaded from strategy
        "sell_pump_threshold_48_1": 0.9,  # value loaded from strategy
        "sell_pump_threshold_48_2": 0.7,  # value loaded from strategy
        "sell_pump_threshold_48_3": 0.5,  # value loaded from strategy
        "sell_rsi_1h_7": 81.7,  # value loaded from strategy
        "sell_rsi_bb_1": 79.5,  # value loaded from strategy
        "sell_rsi_bb_2": 81,  # value loaded from strategy
        "sell_rsi_diff_5": 4.4,  # value loaded from strategy
        "sell_rsi_main_3": 82,  # value loaded from strategy
        "sell_rsi_under_6": 79.0,  # value loaded from strategy
        "sell_trail_down_1": 0.03,  # value loaded from strategy
        "sell_trail_down_2": 0.03,  # value loaded from strategy
        "sell_trail_down_3": 0.05,  # value loaded from strategy
        "sell_trail_profit_max_1": 0.6,  # value loaded from strategy
        "sell_trail_profit_max_2": 0.4,  # value loaded from strategy
        "sell_trail_profit_max_3": 0.2,  # value loaded from strategy
        "sell_trail_profit_min_1": 0.16,  # value loaded from strategy
        "sell_trail_profit_min_2": 0.1,  # value loaded from strategy
        "sell_trail_profit_min_3": 0.06,  # value loaded from strategy
        "sell_trail_rsi_max_1": 50.0,  # value loaded from strategy
        "sell_trail_rsi_max_2": 50.0,  # value loaded from strategy
        "sell_trail_rsi_min_1": 20.0,  # value loaded from strategy
        "sell_trail_rsi_min_2": 20.0,  # value loaded from strategy
    }



    #############################################################
    buy_condition_1_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_01_protection__ema_fast                 = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_01_protection__ema_fast_len             = CategoricalParameter(["26","50","100","200"], default="26", space='buy', optimize=False, load=True)
    buy_01_protection__ema_slow                 = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_01_protection__ema_slow_len             = CategoricalParameter(["26","50","100","200"], default="100", space='buy', optimize=False, load=True)
    buy_01_protection__close_above_ema_fast     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_01_protection__close_above_ema_fast_len = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_01_protection__close_above_ema_slow     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_01_protection__close_above_ema_slow_len = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_01_protection__sma200_rising            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_01_protection__sma200_rising_val        = CategoricalParameter(["20","30","36","44","50"], default="28", space='buy', optimize=False, load=True)
    buy_01_protection__sma200_1h_rising         = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_01_protection__sma200_1h_rising_val     = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_01_protection__safe_dips                = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_01_protection__safe_dips_type           = CategoricalParameter(["10","50","100"], default="80", space='buy', optimize=False, load=True)
    buy_01_protection__safe_pump                = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_01_protection__safe_pump_type           = CategoricalParameter(["10","50","100"], default="70", space='buy', optimize=False, load=True)
    buy_01_protection__safe_pump_period         = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_2_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_02_protection__close_above_ema_fast     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_02_protection__close_above_ema_fast_len = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_02_protection__close_above_ema_slow     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_02_protection__close_above_ema_slow_len = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_02_protection__ema_fast                 = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_02_protection__ema_fast_len             = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_02_protection__ema_slow                 = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_02_protection__ema_slow_len             = CategoricalParameter(["26","50","100","200"], default="20", space='buy', optimize=False, load=True)
    buy_02_protection__safe_dips                = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_02_protection__safe_dips_type           = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_02_protection__safe_pump                = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_02_protection__safe_pump_type           = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=False, load=True)
    buy_02_protection__safe_pump_period         = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)
    buy_02_protection__sma200_rising            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_02_protection__sma200_rising_val        = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_02_protection__sma200_1h_rising         = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_02_protection__sma200_1h_rising_val     = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)

    buy_condition_3_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_03_protection__close_above_ema_fast     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_03_protection__close_above_ema_fast_len = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_03_protection__close_above_ema_slow     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_03_protection__close_above_ema_slow_len = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_03_protection__ema_fast                 = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_03_protection__ema_fast_len             = CategoricalParameter(["26","50","100","200"], default="100", space='buy', optimize=False, load=True)
    buy_03_protection__ema_slow                 = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_03_protection__ema_slow_len             = CategoricalParameter(["26","50","100","200"], default="100", space='buy', optimize=False, load=True)
    buy_03_protection__safe_dips                = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_03_protection__safe_dips_type           = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=False, load=True)
    buy_03_protection__safe_pump                = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_03_protection__safe_pump_type           = CategoricalParameter(["10","50","100"], default="100", space='buy', optimize=False, load=True)
    buy_03_protection__safe_pump_period         = CategoricalParameter(["24","36","48"], default="36", space='buy', optimize=False, load=True)
    buy_03_protection__sma200_rising            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_03_protection__sma200_rising_val        = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_03_protection__sma200_1h_rising         = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_03_protection__sma200_1h_rising_val     = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)

    buy_condition_4_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_04_protection__close_above_ema_fast     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_04_protection__close_above_ema_fast_len = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_04_protection__close_above_ema_slow     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_04_protection__close_above_ema_slow_len = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_04_protection__ema_fast                 = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_04_protection__ema_fast_len             = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_04_protection__ema_slow                 = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_04_protection__ema_slow_len             = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_04_protection__safe_dips                = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_04_protection__safe_dips_type           = CategoricalParameter(["10","50","100"], default="40", space='buy', optimize=False, load=True)
    buy_04_protection__safe_pump                = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_04_protection__safe_pump_type           = CategoricalParameter(["10","50","100"], default="110", space='buy', optimize=False, load=True)
    buy_04_protection__safe_pump_period         = CategoricalParameter(["24","36","48"], default="48", space='buy', optimize=False, load=True)
    buy_04_protection__sma200_rising            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_04_protection__sma200_rising_val        = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_04_protection__sma200_1h_rising         = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_04_protection__sma200_1h_rising_val     = CategoricalParameter(["20","30","36","44","50"], default="20", space='buy', optimize=False, load=True)

    buy_condition_5_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_05_protection__close_above_ema_fast     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_05_protection__close_above_ema_fast_len = CategoricalParameter(["12","20","26","50","100","200"], default="100", space='buy', optimize=False, load=True)
    buy_05_protection__close_above_ema_slow     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_05_protection__close_above_ema_slow_len = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_05_protection__ema_fast                 = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_05_protection__ema_fast_len             = CategoricalParameter(["26","50","100","200"], default="100", space='buy', optimize=False, load=True)
    buy_05_protection__ema_slow                 = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_05_protection__ema_slow_len             = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_05_protection__safe_dips                = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_05_protection__safe_dips_type           = CategoricalParameter(["10","50","100"], default="100", space='buy', optimize=False, load=True)
    buy_05_protection__safe_pump                = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_05_protection__safe_pump_type           = CategoricalParameter(["10","50","100"], default="20", space='buy', optimize=False, load=True)
    buy_05_protection__safe_pump_period         = CategoricalParameter(["24","36","48"], default="36", space='buy', optimize=False, load=True)
    buy_05_protection__sma200_rising            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_05_protection__sma200_rising_val        = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_05_protection__sma200_1h_rising         = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_05_protection__sma200_1h_rising_val     = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)

    buy_condition_6_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_06_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_06_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_06_protection__ema_slow             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_06_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="100", space='buy', optimize=False, load=True)
    buy_06_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_06_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_06_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_06_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_06_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_06_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_06_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_06_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_06_protection__safe_dips            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_06_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_06_protection__safe_pump            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_06_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="20", space='buy', optimize=False, load=True)
    buy_06_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="36", space='buy', optimize=False, load=True)

    buy_condition_7_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_07_protection__ema_fast             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_07_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="100", space='buy', optimize=False, load=True)
    buy_07_protection__ema_slow             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_07_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="12", space='buy', optimize=False, load=True)
    buy_07_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_07_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_07_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_07_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_07_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_07_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_07_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_07_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_07_protection__safe_dips            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_07_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_07_protection__safe_pump            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_07_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_07_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_8_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_08_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_08_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_08_protection__ema_slow             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_08_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="12", space='buy', optimize=False, load=True)
    buy_08_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_08_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_08_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_08_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_08_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_08_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_08_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_08_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_08_protection__safe_dips            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_08_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="100", space='buy', optimize=False, load=True)
    buy_08_protection__safe_pump            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_08_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="120", space='buy', optimize=False, load=True)
    buy_08_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_9_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_09_protection__ema_fast             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_09_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="100", space='buy', optimize=False, load=True)
    buy_09_protection__ema_slow             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_09_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_09_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_09_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_09_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_09_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_09_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_09_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_09_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_09_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_09_protection__safe_dips            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_09_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=False, load=True)
    buy_09_protection__safe_pump            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_09_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_09_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_10_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_10_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_10_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_10_protection__ema_slow             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_10_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_10_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_10_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_10_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_10_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_10_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_10_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_10_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_10_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="24", space='buy', optimize=False, load=True)
    buy_10_protection__safe_dips            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_10_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="100", space='buy', optimize=False, load=True)
    buy_10_protection__safe_pump            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_10_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_10_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_11_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_11_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_11_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_11_protection__ema_slow             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_11_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_11_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_11_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_11_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_11_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_11_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_11_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_11_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_11_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_11_protection__safe_dips            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_11_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="100", space='buy', optimize=False, load=True)
    buy_11_protection__safe_pump            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_11_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_11_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_12_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_12_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_12_protection__ema_slow             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_12_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_12_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_12_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_12_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_12_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="24", space='buy', optimize=False, load=True)
    buy_12_protection__safe_dips            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="100", space='buy', optimize=False, load=True)
    buy_12_protection__safe_pump            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_12_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_12_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_13_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_13_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_13_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_13_protection__ema_slow             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_13_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_13_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_13_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_13_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_13_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_13_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_13_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_13_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_13_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="24", space='buy', optimize=False, load=True)
    buy_13_protection__safe_dips            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_13_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=False, load=True)
    buy_13_protection__safe_pump            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_13_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_13_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_14_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_14_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_14_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_14_protection__ema_slow             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_14_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_14_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_14_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_14_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_14_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_14_protection__sma200_rising        = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_14_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="30", space='buy', optimize=False, load=True)
    buy_14_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_14_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_14_protection__safe_dips            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_14_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=False, load=True)
    buy_14_protection__safe_pump            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_14_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="70", space='buy', optimize=False, load=True)
    buy_14_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_15_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_15_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_15_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_15_protection__ema_slow             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_15_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_15_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_15_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_15_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_15_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_15_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_15_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_15_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_15_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_15_protection__safe_dips            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_15_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_15_protection__safe_pump            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_15_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=False, load=True)
    buy_15_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="36", space='buy', optimize=False, load=True)

    buy_condition_16_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_16_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_16_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_16_protection__ema_slow             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_16_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_16_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_16_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_16_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_16_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="50", space='buy', optimize=False, load=True)
    buy_16_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_16_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_16_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_16_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_16_protection__safe_dips            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_16_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=False, load=True)
    buy_16_protection__safe_pump            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_16_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=False, load=True)
    buy_16_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_17_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_17_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_17_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_17_protection__ema_slow             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_17_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_17_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_17_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_17_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_17_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_17_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_17_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_17_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_17_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_17_protection__safe_dips            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_17_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=False, load=True)
    buy_17_protection__safe_pump            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_17_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="120", space='buy', optimize=False, load=True)
    buy_17_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_18_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_18_protection__ema_fast             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_18_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="100", space='buy', optimize=False, load=True)
    buy_18_protection__ema_slow             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_18_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_18_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_18_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_18_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_18_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_18_protection__sma200_rising        = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_18_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="44", space='buy', optimize=False, load=True)
    buy_18_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_18_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="72", space='buy', optimize=False, load=True)
    buy_18_protection__safe_dips            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_18_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_18_protection__safe_pump            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_18_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="60", space='buy', optimize=False, load=True)
    buy_18_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_19_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_19_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_19_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_19_protection__ema_slow             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_19_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="100", space='buy', optimize=False, load=True)
    buy_19_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_19_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_19_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_19_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_19_protection__sma200_rising        = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_19_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="36", space='buy', optimize=False, load=True)
    buy_19_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_19_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_19_protection__safe_dips            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_19_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_19_protection__safe_pump            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_19_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_19_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_20_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_20_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_20_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_20_protection__ema_slow             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_20_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_20_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_20_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_20_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_20_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_20_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_20_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_20_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_20_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_20_protection__safe_dips            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_20_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=False, load=True)
    buy_20_protection__safe_pump            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_20_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_20_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_21_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_21_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_21_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_21_protection__ema_slow             = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_21_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_21_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_21_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_21_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_21_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_21_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_21_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_21_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_21_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_21_protection__safe_dips            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_21_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="90", space='buy', optimize=False, load=True)
    buy_21_protection__safe_pump            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_21_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_21_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_22_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_22_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_22_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_22_protection__ema_slow             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_22_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_22_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_22_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_22_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_22_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_22_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_22_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_22_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_22_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_22_protection__safe_dips            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_22_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=False, load=True)
    buy_22_protection__safe_pump            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_22_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_22_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_23_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_23_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_23_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_23_protection__ema_slow             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_23_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_23_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_23_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_23_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_23_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_23_protection__sma200_rising        = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_23_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_23_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_23_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=False, load=True)
    buy_23_protection__safe_dips            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_23_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="100", space='buy', optimize=False, load=True)
    buy_23_protection__safe_pump            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_23_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_23_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_24_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_24_protection__ema_fast             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_24_protection__ema_fast_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_24_protection__ema_slow             = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_24_protection__ema_slow_len         = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=False, load=True)
    buy_24_protection__close_above_ema_fast      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_24_protection__close_above_ema_fast_len  = CategoricalParameter(["12","20","26","50","100","200"], default="200", space='buy', optimize=False, load=True)
    buy_24_protection__close_above_ema_slow      = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_24_protection__close_above_ema_slow_len  = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=False, load=True)
    buy_24_protection__sma200_rising        = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_24_protection__sma200_rising_val    = CategoricalParameter(["20","30","36","44","50"], default="30", space='buy', optimize=False, load=True)
    buy_24_protection__sma200_1h_rising     = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_24_protection__sma200_1h_rising_val = CategoricalParameter(["20","30","36","44","50"], default="36", space='buy', optimize=False, load=True)
    buy_24_protection__safe_dips            = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)
    buy_24_protection__safe_dips_type       = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=False, load=True)
    buy_24_protection__safe_pump            = CategoricalParameter([True, False], default=False, space='buy', optimize=False, load=True)
    buy_24_protection__safe_pump_type       = CategoricalParameter(["10","50","100"], default="50", space='buy', optimize=False, load=True)
    buy_24_protection__safe_pump_period     = CategoricalParameter(["24","36","48"], default="24", space='buy', optimize=False, load=True)

    buy_condition_26_enable = CategoricalParameter([True, False], default=True, space='buy', optimize=False, load=True)

    buy_26_protection__optimize = False
    buy_26_parameters__optimize = False

    buy_26_protection__ema_fast                 = CategoricalParameter([True, False], default=False, space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__ema_fast_len             = CategoricalParameter(["26","50","100","200"], default="50", space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__ema_slow                 = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__ema_slow_len             = CategoricalParameter(["26","50","100","200"], default="100", space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__close_above_ema_fast     = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__close_above_ema_fast_len = CategoricalParameter(["12","20","26","50","100","200"], default="50", space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__close_above_ema_slow     = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__close_above_ema_slow_len = CategoricalParameter(["15","50","200"], default="200", space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__sma200_rising            = CategoricalParameter([True, False], default=False, space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__sma200_rising_val        = CategoricalParameter(["20","30","36","44","50"], default="30", space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__sma200_1h_rising         = CategoricalParameter([True, False], default=False, space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__sma200_1h_rising_val     = CategoricalParameter(["20","30","36","44","50"], default="50", space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__safe_dips                = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__safe_dips_type           = CategoricalParameter(["10","50","100"], default="10", space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__safe_pump                = CategoricalParameter([True, False], default=True, space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__safe_pump_type           = CategoricalParameter(["10","50","100"], default="100", space='buy', optimize=buy_26_protection__optimize, load=True)
    buy_26_protection__safe_pump_period         = CategoricalParameter(["24","36","48"], default="48", space='buy', optimize=buy_26_protection__optimize, load=True)

    # Strict dips - level 10
    buy_dip_threshold_10_1 = DecimalParameter(0.001, 0.05, default=0.015, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_10_2 = DecimalParameter(0.01, 0.2, default=0.1, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_10_3 = DecimalParameter(0.1, 0.3, default=0.24, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_10_4 = DecimalParameter(0.3, 0.5, default=0.42, space='buy', decimals=3, optimize=False, load=True)
    # Strict dips - level 20
    buy_dip_threshold_20_1 = DecimalParameter(0.001, 0.05, default=0.016, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_20_2 = DecimalParameter(0.01, 0.2, default=0.11, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_20_3 = DecimalParameter(0.1, 0.4, default=0.26, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_20_4 = DecimalParameter(0.36, 0.56, default=0.44, space='buy', decimals=3, optimize=False, load=True)
    # Strict dips - level 30
    buy_dip_threshold_30_1 = DecimalParameter(0.001, 0.05, default=0.018, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_30_2 = DecimalParameter(0.01, 0.2, default=0.12, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_30_3 = DecimalParameter(0.1, 0.4, default=0.28, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_30_4 = DecimalParameter(0.36, 0.56, default=0.46, space='buy', decimals=3, optimize=False, load=True)
    # Strict dips - level 40
    buy_dip_threshold_40_1 = DecimalParameter(0.001, 0.05, default=0.019, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_40_2 = DecimalParameter(0.01, 0.2, default=0.13, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_40_3 = DecimalParameter(0.1, 0.4, default=0.3, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_40_4 = DecimalParameter(0.36, 0.56, default=0.48, space='buy', decimals=3, optimize=False, load=True)
    # Normal dips - level 50
    buy_dip_threshold_50_1 = DecimalParameter(0.001, 0.05, default=0.02, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_50_2 = DecimalParameter(0.01, 0.2, default=0.14, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_50_3 = DecimalParameter(0.05, 0.4, default=0.32, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_50_4 = DecimalParameter(0.2, 0.5, default=0.5, space='buy', decimals=3, optimize=False, load=True)
    # Normal dips - level 60
    buy_dip_threshold_60_1 = DecimalParameter(0.001, 0.05, default=0.022, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_60_2 = DecimalParameter(0.1, 0.22, default=0.18, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_60_3 = DecimalParameter(0.2, 0.4, default=0.34, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_60_4 = DecimalParameter(0.4, 0.6, default=0.56, space='buy', decimals=3, optimize=False, load=True)
    # Normal dips - level 70
    buy_dip_threshold_70_1 = DecimalParameter(0.001, 0.05, default=0.023, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_70_2 = DecimalParameter(0.16, 0.28, default=0.2, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_70_3 = DecimalParameter(0.2, 0.4, default=0.36, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_70_4 = DecimalParameter(0.5, 0.7, default=0.6, space='buy', decimals=3, optimize=False, load=True)
    # Normal dips - level 80
    buy_dip_threshold_80_1 = DecimalParameter(0.001, 0.05, default=0.024, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_80_2 = DecimalParameter(0.16, 0.28, default=0.22, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_80_3 = DecimalParameter(0.2, 0.4, default=0.38, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_80_4 = DecimalParameter(0.5, 0.7, default=0.66, space='buy', decimals=3, optimize=False, load=True)
    # Normal dips - level 70
    buy_dip_threshold_90_1 = DecimalParameter(0.001, 0.05, default=0.025, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_90_2 = DecimalParameter(0.16, 0.28, default=0.23, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_90_3 = DecimalParameter(0.3, 0.5, default=0.4, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_90_4 = DecimalParameter(0.6, 0.8, default=0.7, space='buy', decimals=3, optimize=False, load=True)
    # Loose dips - level 100
    buy_dip_threshold_100_1 = DecimalParameter(0.001, 0.05, default=0.026, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_100_2 = DecimalParameter(0.16, 0.3, default=0.24, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_100_3 = DecimalParameter(0.3, 0.5, default=0.42, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_100_4 = DecimalParameter(0.6, 1.0, default=0.8, space='buy', decimals=3, optimize=False, load=True)
    # Loose dips - level 110
    buy_dip_threshold_110_1 = DecimalParameter(0.001, 0.05, default=0.027, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_110_2 = DecimalParameter(0.16, 0.3, default=0.26, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_110_3 = DecimalParameter(0.3, 0.5, default=0.44, space='buy', decimals=3, optimize=False, load=True)
    buy_dip_threshold_110_4 = DecimalParameter(0.6, 1.0, default=0.84, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 10
    buy_pump_pull_threshold_10_24 = DecimalParameter(1.5, 3.0, default=2.2, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_10_24 = DecimalParameter(0.4, 1.0, default=0.42, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 10
    buy_pump_pull_threshold_10_36 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_10_36 = DecimalParameter(0.4, 1.0, default=0.58, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 10
    buy_pump_pull_threshold_10_48 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_10_48 = DecimalParameter(0.4, 1.0, default=0.8, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 20
    buy_pump_pull_threshold_20_24 = DecimalParameter(1.5, 3.0, default=2.2, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_20_24 = DecimalParameter(0.4, 1.0, default=0.46, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 20
    buy_pump_pull_threshold_20_36 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_20_36 = DecimalParameter(0.4, 1.0, default=0.6, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 20
    buy_pump_pull_threshold_20_48 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_20_48 = DecimalParameter(0.4, 1.0, default=0.81, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 30
    buy_pump_pull_threshold_30_24 = DecimalParameter(1.5, 3.0, default=2.2, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_30_24 = DecimalParameter(0.4, 1.0, default=0.5, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 30
    buy_pump_pull_threshold_30_36 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_30_36 = DecimalParameter(0.4, 1.0, default=0.62, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 30
    buy_pump_pull_threshold_30_48 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_30_48 = DecimalParameter(0.4, 1.0, default=0.82, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 40
    buy_pump_pull_threshold_40_24 = DecimalParameter(1.5, 3.0, default=2.2, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_40_24 = DecimalParameter(0.4, 1.0, default=0.54, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 40
    buy_pump_pull_threshold_40_36 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_40_36 = DecimalParameter(0.4, 1.0, default=0.63, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 40
    buy_pump_pull_threshold_40_48 = DecimalParameter(1.5, 3.0, default=2.0, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_40_48 = DecimalParameter(0.4, 1.0, default=0.84, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 50
    buy_pump_pull_threshold_50_24 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_50_24 = DecimalParameter(0.4, 1.0, default=0.6, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 50
    buy_pump_pull_threshold_50_36 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_50_36 = DecimalParameter(0.4, 1.0, default=0.64, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 50
    buy_pump_pull_threshold_50_48 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_50_48 = DecimalParameter(0.4, 1.0, default=0.85, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 60
    buy_pump_pull_threshold_60_24 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_60_24 = DecimalParameter(0.4, 1.0, default=0.62, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 60
    buy_pump_pull_threshold_60_36 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_60_36 = DecimalParameter(0.4, 1.0, default=0.66, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 60
    buy_pump_pull_threshold_60_48 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_60_48 = DecimalParameter(0.4, 1.0, default=0.9, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 70
    buy_pump_pull_threshold_70_24 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_70_24 = DecimalParameter(0.4, 1.0, default=0.63, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 70
    buy_pump_pull_threshold_70_36 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_70_36 = DecimalParameter(0.4, 1.0, default=0.67, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 70
    buy_pump_pull_threshold_70_48 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_70_48 = DecimalParameter(0.4, 1.0, default=0.95, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 80
    buy_pump_pull_threshold_80_24 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_80_24 = DecimalParameter(0.4, 1.0, default=0.64, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 80
    buy_pump_pull_threshold_80_36 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_80_36 = DecimalParameter(0.4, 1.0, default=0.68, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 80
    buy_pump_pull_threshold_80_48 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_80_48 = DecimalParameter(0.8, 1.1, default=1.0, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 90
    buy_pump_pull_threshold_90_24 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_90_24 = DecimalParameter(0.4, 1.0, default=0.65, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 90
    buy_pump_pull_threshold_90_36 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_90_36 = DecimalParameter(0.4, 1.0, default=0.69, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 90
    buy_pump_pull_threshold_90_48 = DecimalParameter(1.5, 3.0, default=1.75, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_90_48 = DecimalParameter(0.8, 1.2, default=1.1, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 100
    buy_pump_pull_threshold_100_24 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_100_24 = DecimalParameter(0.4, 1.0, default=0.66, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 100
    buy_pump_pull_threshold_100_36 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_100_36 = DecimalParameter(0.4, 1.0, default=0.7, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 100
    buy_pump_pull_threshold_100_48 = DecimalParameter(1.3, 2.0, default=1.4, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_100_48 = DecimalParameter(0.4, 1.8, default=1.6, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 110
    buy_pump_pull_threshold_110_24 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_110_24 = DecimalParameter(0.4, 1.0, default=0.7, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 110
    buy_pump_pull_threshold_110_36 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_110_36 = DecimalParameter(0.4, 1.0, default=0.74, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 110
    buy_pump_pull_threshold_110_48 = DecimalParameter(1.3, 2.0, default=1.4, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_110_48 = DecimalParameter(1.4, 2.0, default=1.8, space='buy', decimals=3, optimize=False, load=True)

    # 24 hours - level 120
    buy_pump_pull_threshold_120_24 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_120_24 = DecimalParameter(0.4, 1.0, default=0.78, space='buy', decimals=3, optimize=False, load=True)
    # 36 hours - level 120
    buy_pump_pull_threshold_120_36 = DecimalParameter(1.5, 3.0, default=1.7, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_120_36 = DecimalParameter(0.4, 1.0, default=0.78, space='buy', decimals=3, optimize=False, load=True)
    # 48 hours - level 120
    buy_pump_pull_threshold_120_48 = DecimalParameter(1.3, 2.0, default=1.4, space='buy', decimals=2, optimize=False, load=True)
    buy_pump_threshold_120_48 = DecimalParameter(1.4, 2.8, default=2.0, space='buy', decimals=3, optimize=False, load=True)

    buy_min_inc_1 = DecimalParameter(0.01, 0.05, default=0.022, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_1 = DecimalParameter(25.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_1 = DecimalParameter(70.0, 90.0, default=84.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1 = DecimalParameter(20.0, 40.0, default=36.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_1 = DecimalParameter(20.0, 40.0, default=36.0, space='buy', decimals=1, optimize=False, load=True)

    buy_rsi_1h_min_2 = DecimalParameter(30.0, 40.0, default=32.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_2 = DecimalParameter(70.0, 95.0, default=84.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_diff_2 = DecimalParameter(30.0, 50.0, default=39.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_2 = DecimalParameter(30.0, 56.0, default=49.0, space='buy', decimals=1, optimize=False, load=True)
    buy_bb_offset_2 = DecimalParameter(0.97, 0.999, default=0.983, space='buy', decimals=3, optimize=False, load=True)

    buy_bb40_bbdelta_close_3 = DecimalParameter(0.005, 0.06, default=0.059, space='buy', optimize=False, load=True)
    buy_bb40_closedelta_close_3 = DecimalParameter(0.01, 0.03, default=0.023, space='buy', optimize=False, load=True)
    buy_bb40_tail_bbdelta_3 = DecimalParameter(0.15, 0.45, default=0.418, space='buy', optimize=False, load=True)
    buy_ema_rel_3 = DecimalParameter(0.97, 0.999, default=0.986, space='buy', decimals=3, optimize=False, load=True)

    buy_bb20_close_bblowerband_4 = DecimalParameter(0.96, 0.99, default=0.98, space='buy', optimize=False, load=True)
    buy_bb20_volume_4 = DecimalParameter(1.0, 20.0, default=10.0, space='buy', decimals=2, optimize=False, load=True)

    buy_ema_open_mult_5 = DecimalParameter(0.016, 0.03, default=0.018, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_5 = DecimalParameter(0.98, 1.0, default=0.996, space='buy', decimals=3, optimize=False, load=True)
    buy_ema_rel_5 = DecimalParameter(0.97, 0.999, default=0.944, space='buy', decimals=3, optimize=False, load=True)

    buy_ema_open_mult_6 = DecimalParameter(0.02, 0.03, default=0.021, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_6 = DecimalParameter(0.98, 0.999, default=0.984, space='buy', decimals=3, optimize=False, load=True)

    buy_ema_open_mult_7 = DecimalParameter(0.02, 0.04, default=0.03, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_7 = DecimalParameter(24.0, 50.0, default=37.0, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_8 = DecimalParameter(1.0, 6.0, default=2.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_8 = DecimalParameter(16.0, 30.0, default=29.0, space='buy', decimals=1, optimize=False, load=True)
    buy_tail_diff_8 = DecimalParameter(3.0, 10.0, default=3.5, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_9 = DecimalParameter(0.91, 0.94, default=0.922, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_9 = DecimalParameter(0.96, 0.98, default=0.942, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_9 = DecimalParameter(26.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_9 = DecimalParameter(70.0, 90.0, default=88.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_9 = DecimalParameter(36.0, 56.0, default=50.0, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_10 = DecimalParameter(0.93, 0.97, default=0.948, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_10 = DecimalParameter(0.97, 0.99, default=0.985, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_10 = DecimalParameter(20.0, 40.0, default=37.0, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_11 = DecimalParameter(0.93, 0.99, default=0.934, space='buy', decimals=3, optimize=False, load=True)
    buy_min_inc_11 = DecimalParameter(0.005, 0.05, default=0.01, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_1h_min_11 = DecimalParameter(40.0, 60.0, default=55.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_max_11 = DecimalParameter(70.0, 90.0, default=84.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_11 = DecimalParameter(34.0, 50.0, default=48.0, space='buy', decimals=1, optimize=False, load=True)
    buy_mfi_11 = DecimalParameter(30.0, 46.0, default=36.0, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_12 = DecimalParameter(0.93, 0.97, default=0.922, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_12 = DecimalParameter(26.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ewo_12 = DecimalParameter(1.0, 6.0, default=1.8, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_13 = DecimalParameter(0.93, 0.98, default=0.99, space='buy', decimals=3, optimize=False, load=True)
    buy_ewo_13 = DecimalParameter(-14.0, -7.0, default=-11.4, space='buy', decimals=1, optimize=False, load=True)

    buy_ema_open_mult_14 = DecimalParameter(0.01, 0.03, default=0.014, space='buy', decimals=3, optimize=False, load=True)
    buy_bb_offset_14 = DecimalParameter(0.98, 1.0, default=0.988, space='buy', decimals=3, optimize=False, load=True)
    buy_ma_offset_14 = DecimalParameter(0.93, 0.99, default=0.98, space='buy', decimals=3, optimize=False, load=True)

    buy_ema_open_mult_15 = DecimalParameter(0.01, 0.03, default=0.018, space='buy', decimals=3, optimize=False, load=True)
    buy_ma_offset_15 = DecimalParameter(0.93, 0.99, default=0.954, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_15 = DecimalParameter(20.0, 36.0, default=28.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ema_rel_15 = DecimalParameter(0.97, 0.999, default=0.988, space='buy', decimals=3, optimize=False, load=True)

    buy_ma_offset_16 = DecimalParameter(0.93, 0.97, default=0.952, space='buy', decimals=3, optimize=False, load=True)
    buy_rsi_16 = DecimalParameter(26.0, 50.0, default=31.0, space='buy', decimals=1, optimize=False, load=True)
    buy_ewo_16 = DecimalParameter(2.0, 6.0, default=2.8, space='buy', decimals=1, optimize=False, load=True)

    buy_ma_offset_17 = DecimalParameter(0.93, 0.98, default=0.952, space='buy', decimals=3, optimize=False, load=True)
    buy_ewo_17 = DecimalParameter(-18.0, -10.0, default=-12.8, space='buy', decimals=1, optimize=False, load=True)

    buy_rsi_18 = DecimalParameter(16.0, 32.0, default=26.0, space='buy', decimals=1, optimize=False, load=True)
    buy_bb_offset_18 = DecimalParameter(0.98, 1.0, default=0.982, space='buy', decimals=3, optimize=False, load=True)

    buy_rsi_1h_min_19 = DecimalParameter(40.0, 70.0, default=50.0, space='buy', decimals=1, optimize=False, load=True)
    buy_chop_min_19 = DecimalParameter(20.0, 60.0, default=24.1, space='buy', decimals=1, optimize=False, load=True)

    buy_rsi_20 = DecimalParameter(20.0, 36.0, default=27.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_20 = DecimalParameter(14.0, 30.0, default=20.0, space='buy', decimals=1, optimize=False, load=True)

    buy_rsi_21 = DecimalParameter(10.0, 28.0, default=23.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_21 = DecimalParameter(18.0, 40.0, default=24.0, space='buy', decimals=1, optimize=False, load=True)

    buy_volume_22 = DecimalParameter(0.5, 6.0, default=3.0, space='buy', decimals=1, optimize=False, load=True)
    buy_bb_offset_22 = DecimalParameter(0.98, 1.0, default=0.98, space='buy', decimals=3, optimize=False, load=True)
    buy_ma_offset_22 = DecimalParameter(0.93, 0.98, default=0.94, space='buy', decimals=3, optimize=False, load=True)
    buy_ewo_22 = DecimalParameter(2.0, 10.0, default=4.2, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_22 = DecimalParameter(26.0, 56.0, default=37.0, space='buy', decimals=1, optimize=False, load=True)

    buy_bb_offset_23 = DecimalParameter(0.97, 1.0, default=0.987, space='buy', decimals=3, optimize=False, load=True)
    buy_ewo_23 = DecimalParameter(2.0, 10.0, default=7.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_23 = DecimalParameter(20.0, 40.0, default=30.0, space='buy', decimals=1, optimize=False, load=True)
    buy_rsi_1h_23 = DecimalParameter(60.0, 80.0, default=70.0, space='buy', decimals=1, optimize=False, load=True)

    buy_24_rsi_max = DecimalParameter(26.0, 60.0, default=60.0, space='buy', decimals=1, optimize=False, load=True)
    buy_24_rsi_1h_min = DecimalParameter(40.0, 90.0, default=66.9, space='buy', decimals=1, optimize=False, load=True)

    buy_26_zema_low_offset = DecimalParameter(0.90, 0.99, default=0.93, space='buy', optimize=buy_26_parameters__optimize, load=True)

    # Sell

    sell_condition_1_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_2_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_3_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_4_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_5_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_6_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_7_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)
    sell_condition_8_enable = CategoricalParameter([True, False], default=True, space='sell', optimize=False, load=True)

    # 48h for pump sell checks
    sell_pump_threshold_48_1 = DecimalParameter(0.5, 1.2, default=0.9, space='sell', decimals=2, optimize=False, load=True)
    sell_pump_threshold_48_2 = DecimalParameter(0.4, 0.9, default=0.7, space='sell', decimals=2, optimize=False, load=True)
    sell_pump_threshold_48_3 = DecimalParameter(0.3, 0.7, default=0.5, space='sell', decimals=2, optimize=False, load=True)

    # 36h for pump sell checks
    sell_pump_threshold_36_1 = DecimalParameter(0.5, 0.9, default=0.72, space='sell', decimals=2, optimize=False, load=True)
    sell_pump_threshold_36_2 = DecimalParameter(3.0, 6.0, default=4.0, space='sell', decimals=2, optimize=False, load=True)
    sell_pump_threshold_36_3 = DecimalParameter(0.8, 1.6, default=1.0, space='sell', decimals=2, optimize=False, load=True)

    # 24h for pump sell checks
    sell_pump_threshold_24_1 = DecimalParameter(0.5, 0.9, default=0.68, space='sell', decimals=2, optimize=False, load=True)
    sell_pump_threshold_24_2 = DecimalParameter(0.3, 0.6, default=0.62, space='sell', decimals=2, optimize=False, load=True)
    sell_pump_threshold_24_3 = DecimalParameter(0.2, 0.5, default=0.88, space='sell', decimals=2, optimize=False, load=True)

    sell_rsi_bb_1 = DecimalParameter(60.0, 80.0, default=79.5, space='sell', decimals=1, optimize=False, load=True)

    sell_rsi_bb_2 = DecimalParameter(72.0, 90.0, default=81, space='sell', decimals=1, optimize=False, load=True)

    sell_rsi_main_3 = DecimalParameter(77.0, 90.0, default=82, space='sell', decimals=1, optimize=False, load=True)

    sell_dual_rsi_rsi_4 = DecimalParameter(72.0, 84.0, default=73.4, space='sell', decimals=1, optimize=False, load=True)
    sell_dual_rsi_rsi_1h_4 = DecimalParameter(78.0, 92.0, default=79.6, space='sell', decimals=1, optimize=False, load=True)

    sell_ema_relative_5 = DecimalParameter(0.005, 0.05, default=0.024, space='sell', optimize=False, load=True)
    sell_rsi_diff_5 = DecimalParameter(0.0, 20.0, default=4.4, space='sell', optimize=False, load=True)

    sell_rsi_under_6 = DecimalParameter(72.0, 90.0, default=79.0, space='sell', decimals=1, optimize=False, load=True)

    sell_rsi_1h_7 = DecimalParameter(80.0, 95.0, default=81.7, space='sell', decimals=1, optimize=False, load=True)

    sell_bb_relative_8 = DecimalParameter(1.05, 1.3, default=1.1, space='sell', decimals=3, optimize=False, load=True)

    sell_custom_profit_0 = DecimalParameter(0.01, 0.1, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_0 = DecimalParameter(30.0, 40.0, default=34.0, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_profit_1 = DecimalParameter(0.01, 0.1, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_1 = DecimalParameter(30.0, 50.0, default=35.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_2 = DecimalParameter(0.01, 0.1, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_2 = DecimalParameter(30.0, 50.0, default=37.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_3 = DecimalParameter(0.01, 0.1, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_3 = DecimalParameter(30.0, 50.0, default=42.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_4 = DecimalParameter(0.01, 0.1, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_4 = DecimalParameter(35.0, 50.0, default=43.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_5 = DecimalParameter(0.01, 0.1, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_5 = DecimalParameter(35.0, 50.0, default=45.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_6 = DecimalParameter(0.01, 0.1, default=0.07, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_6 = DecimalParameter(38.0, 55.0, default=48.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_7 = DecimalParameter(0.01, 0.1, default=0.08, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_7 = DecimalParameter(40.0, 58.0, default=54.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_8 = DecimalParameter(0.06, 0.1, default=0.09, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_8 = DecimalParameter(40.0, 50.0, default=55.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_9 = DecimalParameter(0.05, 0.14, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_9 = DecimalParameter(40.0, 60.0, default=54.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_10 = DecimalParameter(0.1, 0.14, default=0.12, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_10 = DecimalParameter(38.0, 50.0, default=42.0, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_profit_11 = DecimalParameter(0.16, 0.45, default=0.20, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_rsi_11 = DecimalParameter(28.0, 40.0, default=34.0, space='sell', decimals=2, optimize=False, load=True)

    # Profit under EMA200
    sell_custom_under_profit_0 = DecimalParameter(0.01, 0.4, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_0 = DecimalParameter(28.0, 40.0, default=35.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_1 = DecimalParameter(0.01, 0.10, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_1 = DecimalParameter(36.0, 60.0, default=56.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_2 = DecimalParameter(0.01, 0.10, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_2 = DecimalParameter(46.0, 66.0, default=57.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_3 = DecimalParameter(0.01, 0.10, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_3 = DecimalParameter(50.0, 68.0, default=58.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_4 = DecimalParameter(0.02, 0.1, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_4 = DecimalParameter(50.0, 68.0, default=59.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_5 = DecimalParameter(0.02, 0.1, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_5 = DecimalParameter(46.0, 62.0, default=60.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_6 = DecimalParameter(0.03, 0.1, default=0.07, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_6 = DecimalParameter(44.0, 60.0, default=56.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_7 = DecimalParameter(0.04, 0.1, default=0.08, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_7 = DecimalParameter(46.0, 60.0, default=54.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_8 = DecimalParameter(0.06, 0.12, default=0.09, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_8 = DecimalParameter(40.0, 58.0, default=55.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_9 = DecimalParameter(0.08, 0.14, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_9 = DecimalParameter(40.0, 60.0, default=54.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_10 = DecimalParameter(0.1, 0.16, default=0.12, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_10 = DecimalParameter(30.0, 50.0, default=42.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_under_profit_11 = DecimalParameter(0.16, 0.3, default=0.2, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_under_rsi_11 = DecimalParameter(24.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)

    # Profit targets for pumped pairs 48h 1
    sell_custom_pump_profit_1_1 = DecimalParameter(0.01, 0.03, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_1_1 = DecimalParameter(26.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_1_2 = DecimalParameter(0.01, 0.6, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_1_2 = DecimalParameter(36.0, 50.0, default=40.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_1_3 = DecimalParameter(0.02, 0.10, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_1_3 = DecimalParameter(38.0, 50.0, default=42.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_1_4 = DecimalParameter(0.06, 0.12, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_1_4 = DecimalParameter(36.0, 48.0, default=42.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_1_5 = DecimalParameter(0.14, 0.24, default=0.2, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_1_5 = DecimalParameter(20.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)

    # Profit targets for pumped pairs 36h 1
    sell_custom_pump_profit_2_1 = DecimalParameter(0.01, 0.03, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_2_1 = DecimalParameter(26.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_2_2 = DecimalParameter(0.01, 0.6, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_2_2 = DecimalParameter(36.0, 50.0, default=40.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_2_3 = DecimalParameter(0.02, 0.10, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_2_3 = DecimalParameter(38.0, 50.0, default=40.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_2_4 = DecimalParameter(0.06, 0.12, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_2_4 = DecimalParameter(36.0, 48.0, default=42.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_2_5 = DecimalParameter(0.14, 0.24, default=0.2, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_2_5 = DecimalParameter(20.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)

    # Profit targets for pumped pairs 24h 1
    sell_custom_pump_profit_3_1 = DecimalParameter(0.01, 0.03, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_3_1 = DecimalParameter(26.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_3_2 = DecimalParameter(0.01, 0.6, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_3_2 = DecimalParameter(34.0, 50.0, default=40.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_3_3 = DecimalParameter(0.02, 0.10, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_3_3 = DecimalParameter(38.0, 50.0, default=40.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_3_4 = DecimalParameter(0.06, 0.12, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_3_4 = DecimalParameter(36.0, 48.0, default=42.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_profit_3_5 = DecimalParameter(0.14, 0.24, default=0.2, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_rsi_3_5 = DecimalParameter(20.0, 40.0, default=34.0, space='sell', decimals=1, optimize=False, load=True)

    # SMA descending
    sell_custom_dec_profit_min_1 = DecimalParameter(0.01, 0.10, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_dec_profit_max_1 = DecimalParameter(0.06, 0.16, default=0.12, space='sell', decimals=3, optimize=False, load=True)

    # Under EMA100
    sell_custom_dec_profit_min_2 = DecimalParameter(0.05, 0.12, default=0.07, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_dec_profit_max_2 = DecimalParameter(0.06, 0.2, default=0.16, space='sell', decimals=3, optimize=False, load=True)

    # Trail 1
    sell_trail_profit_min_1 = DecimalParameter(0.1, 0.2, default=0.16, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_profit_max_1 = DecimalParameter(0.4, 0.7, default=0.6, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_1 = DecimalParameter(0.01, 0.08, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_rsi_min_1 = DecimalParameter(16.0, 36.0, default=20.0, space='sell', decimals=1, optimize=False, load=True)
    sell_trail_rsi_max_1 = DecimalParameter(30.0, 50.0, default=50.0, space='sell', decimals=1, optimize=False, load=True)

    # Trail 2
    sell_trail_profit_min_2 = DecimalParameter(0.08, 0.16, default=0.1, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_2 = DecimalParameter(0.3, 0.5, default=0.4, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_2 = DecimalParameter(0.02, 0.08, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_rsi_min_2 = DecimalParameter(16.0, 36.0, default=20.0, space='sell', decimals=1, optimize=False, load=True)
    sell_trail_rsi_max_2 = DecimalParameter(30.0, 50.0, default=50.0, space='sell', decimals=1, optimize=False, load=True)

    # Trail 3
    sell_trail_profit_min_3 = DecimalParameter(0.01, 0.12, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_trail_profit_max_3 = DecimalParameter(0.1, 0.3, default=0.2, space='sell', decimals=2, optimize=False, load=True)
    sell_trail_down_3 = DecimalParameter(0.01, 0.06, default=0.05, space='sell', decimals=3, optimize=False, load=True)

    # Under & near EMA200, accept profit
    sell_custom_profit_under_rel_1 = DecimalParameter(0.01, 0.04, default=0.024, space='sell', optimize=False, load=True)
    sell_custom_profit_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=4.4, space='sell', optimize=False, load=True)

    # Under & near EMA200, take the loss
    sell_custom_stoploss_under_rel_1 = DecimalParameter(0.001, 0.02, default=0.004, space='sell', optimize=False, load=True)
    sell_custom_stoploss_under_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=10.0, space='sell', optimize=False, load=True)

    # Long duration/recover stoploss 1
    sell_custom_stoploss_long_profit_min_1 = DecimalParameter(-0.1, -0.02, default=-0.08, space='sell', optimize=False, load=True)
    sell_custom_stoploss_long_profit_max_1 = DecimalParameter(-0.06, -0.01, default=-0.04, space='sell', optimize=False, load=True)
    sell_custom_stoploss_long_recover_1 = DecimalParameter(0.05, 0.15, default=0.1, space='sell', optimize=False, load=True)
    sell_custom_stoploss_long_rsi_diff_1 = DecimalParameter(0.0, 20.0, default=4.0, space='sell', optimize=False, load=True)

    # Long duration/recover stoploss 2
    sell_custom_stoploss_long_recover_2 = DecimalParameter(0.03, 0.15, default=0.06, space='sell', optimize=False, load=True)
    sell_custom_stoploss_long_rsi_diff_2 = DecimalParameter(30.0, 50.0, default=40.0, space='sell', optimize=False, load=True)

    # Pumped, descending SMA
    sell_custom_pump_dec_profit_min_1 = DecimalParameter(0.001, 0.04, default=0.005, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_max_1 = DecimalParameter(0.03, 0.08, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_min_2 = DecimalParameter(0.01, 0.08, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_max_2 = DecimalParameter(0.04, 0.1, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_min_3 = DecimalParameter(0.02, 0.1, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_max_3 = DecimalParameter(0.06, 0.12, default=0.09, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_min_4 = DecimalParameter(0.01, 0.05, default=0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_dec_profit_max_4 = DecimalParameter(0.02, 0.1, default=0.04, space='sell', decimals=3, optimize=False, load=True)

    # Pumped 48h 1, under EMA200
    sell_custom_pump_under_profit_min_1 = DecimalParameter(0.02, 0.06, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_under_profit_max_1 = DecimalParameter(0.04, 0.1, default=0.09, space='sell', decimals=3, optimize=False, load=True)

    # Pumped trail 1
    sell_custom_pump_trail_profit_min_1 = DecimalParameter(0.01, 0.12, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_trail_profit_max_1 = DecimalParameter(0.06, 0.16, default=0.07, space='sell', decimals=2, optimize=False, load=True)
    sell_custom_pump_trail_down_1 = DecimalParameter(0.01, 0.06, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_pump_trail_rsi_min_1 = DecimalParameter(16.0, 36.0, default=20.0, space='sell', decimals=1, optimize=False, load=True)
    sell_custom_pump_trail_rsi_max_1 = DecimalParameter(30.0, 50.0, default=70.0, space='sell', decimals=1, optimize=False, load=True)

    # Stoploss, pumped, 48h 1
    sell_custom_stoploss_pump_max_profit_1 = DecimalParameter(0.01, 0.04, default=0.025, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_min_1 = DecimalParameter(-0.1, -0.01, default=-0.02, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_max_1 = DecimalParameter(-0.1, -0.01, default=-0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_ma_offset_1 = DecimalParameter(0.7, 0.99, default=0.94, space='sell', decimals=2, optimize=False, load=True)

    # Stoploss, pumped, 48h 1
    sell_custom_stoploss_pump_max_profit_2 = DecimalParameter(0.01, 0.04, default=0.025, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_loss_2 = DecimalParameter(-0.1, -0.01, default=-0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_ma_offset_2 = DecimalParameter(0.7, 0.99, default=0.92, space='sell', decimals=2, optimize=False, load=True)

    # Stoploss, pumped, 36h 3
    sell_custom_stoploss_pump_max_profit_3 = DecimalParameter(0.01, 0.04, default=0.008, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_loss_3 = DecimalParameter(-0.16, -0.06, default=-0.12, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_stoploss_pump_ma_offset_3 = DecimalParameter(0.7, 0.99, default=0.88, space='sell', decimals=2, optimize=False, load=True)

    # Recover
    sell_custom_recover_profit_1 = DecimalParameter(0.01, 0.06, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_recover_min_loss_1 = DecimalParameter(0.06, 0.16, default=0.12, space='sell', decimals=3, optimize=False, load=True)

    sell_custom_recover_profit_min_2 = DecimalParameter(0.01, 0.04, default=0.01, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_recover_profit_max_2 = DecimalParameter(0.02, 0.08, default=0.05, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_recover_min_loss_2 = DecimalParameter(0.04, 0.16, default=0.06, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_recover_rsi_2 = DecimalParameter(32.0, 52.0, default=46.0, space='sell', decimals=1, optimize=False, load=True)

    # Profit for long duration trades
    sell_custom_long_profit_min_1 = DecimalParameter(0.01, 0.04, default=0.03, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_long_profit_max_1 = DecimalParameter(0.02, 0.08, default=0.04, space='sell', decimals=3, optimize=False, load=True)
    sell_custom_long_duration_min_1 = IntParameter(700, 2000, default=900, space='sell', optimize=False, load=True)
    
    #mixed values, mainly from Offset strategy
    ma_type_index_1  = IntParameter(0, 3, default=0, space='buy', optimize=False)
    ma_type_index_2  = IntParameter(0, 3, default=0, space='buy', optimize=False)
    ma_type_index_3  = IntParameter(0, 3, default=0, space='buy', optimize=False)
    ma_type_index_4  = IntParameter(0, 2, default=0, space='sell', optimize=False)
    ma_type_index_5  = IntParameter(0, 2, default=0, space='sell', optimize=False)
    ma_type_index_6  = IntParameter(0, 2, default=0, space='sell', optimize=False)



    buy_bullish_trigger = CategoricalParameter(ma_types.keys(), default='hull', space='buy', optimize=False)
    buy_bearish_trigger = CategoricalParameter(ma_types.keys(), default='zlema', space='buy', optimize=False)


    sell_bullish_trigger = CategoricalParameter(ma_types.keys(), default='hull', space='sell', optimize=False)
    sell_bearish_trigger = CategoricalParameter(ma_types.keys(), default='zlema', space='sell', optimize=False)



    base_nb_candles_bearish_buy = IntParameter(5, 80, default=20, space='buy')
    base_nb_candles_bullish_buy = IntParameter(5, 80, default=20, space='buy')

    base_nb_candles_bullish_sell = IntParameter(10, 30, default=15, space='sell', optimize=True)
    base_nb_candles_bearish_sell = IntParameter(10, 30, default=15, space='sell', optimize=True)


    low_offset_bullish = DecimalParameter(0.95, 1.01, default=0.958, space='buy')
    low_offset_bearish = DecimalParameter(0.925, 0.98, default=0.93, space='buy')

    high_offset_bullish = DecimalParameter(1.01, 1.06, default=1.012, space='sell', optimize=True)
    high_offset_bearish = DecimalParameter(0.98, 1.04, default=0.98, space='sell', optimize=True)

    
    pmaxmultiplier = IntParameter(10, 40, default=10, space='buy', optimize=True)
    pmaxlength  = IntParameter(5, 30, default=9, space='buy', optimize=False)
    atr_multiplier = DecimalParameter(1.0, 4.0, default=3.0, space='sell', decimals=2, optimize=True, load=True)
    atr_rolling_length = IntParameter(1, 15, default=10, space='sell', optimize=True)

    # Protection
    ewo_low = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(2.0, 12.0, default=buy_params['ewo_high'], space='buy', optimize=True)
    fast_ewo = IntParameter(10, 50, default=buy_params['fast_ewo'], space='buy', optimize=False)
    slow_ewo = IntParameter(100, 200, default=buy_params['slow_ewo'], space='buy', optimize=False)
    rsi_buy_bullish = IntParameter(30, 70, default=50, space='buy', optimize=True, load=True)
    rsi_buy_bearish = IntParameter(30, 70, default=50, space='buy', optimize=True, load=True)

    #############################################################

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, **kwargs) -> bool:

        #init struct on buy

        if not pair in self.custom_info:
           self.custom_info[pair] = {}
        self.custom_info[pair]['MA_buy'] = 0
        self.custom_info[pair]['maxProfit']  = 0

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if      ((
		(last_candle[f'pm_{pmaxperiod}_{self.pmaxmultiplier.value}_{self.pmaxlength.value}_{pmaxtype}'] <= last_candle[f'mavalue{self.pmaxlength.value}']) &
                (last_candle['close'] < last_candle[f'{self.ma_type_index_1.value}_ma_buy_bullish{self.base_nb_candles_bullish_buy.value}']* self.low_offset_bullish.value) &
                (last_candle['EWO'] > self.ewo_high.value) &
                (last_candle['rsi14'] < self.rsi_buy_bullish.value) &
                (last_candle['volume'] > 0)
                ) or
                (
		(last_candle[f'pm_{pmaxperiod}_{self.pmaxmultiplier.value}_{self.pmaxlength.value}_{pmaxtype}'] <= last_candle[f'mavalue{self.pmaxlength.value}']) &
                (last_candle['close'] < last_candle[f'{self.ma_type_index_1.value}_ma_buy_bullish{self.base_nb_candles_bullish_buy.value}']* self.low_offset_bullish.value) &
                (last_candle['EWO'] < self.ewo_low.value) &
                (last_candle['volume'] > 0)
                ) or
                (
		(last_candle[f'pm_{pmaxperiod}_{self.pmaxmultiplier.value}_{self.pmaxlength.value}_{pmaxtype}'] > last_candle[f'mavalue{self.pmaxlength.value}']) &
                (last_candle['close'] < last_candle[f'{self.ma_type_index_2.value}_ma_buy_bearish{self.base_nb_candles_bearish_buy.value}']* self.low_offset_bearish.value) &
                (last_candle['EWO'] > self.ewo_high.value) &
                (last_candle['rsi14'] < self.rsi_buy_bearish.value) &
                (last_candle['volume'] > 0)
                ) or
                (
		(last_candle[f'pm_{pmaxperiod}_{self.pmaxmultiplier.value}_{self.pmaxlength.value}_{pmaxtype}'] > last_candle[f'mavalue{self.pmaxlength.value}']) &
                (last_candle['close'] < last_candle[f'{self.ma_type_index_2.value}_ma_buy_bearish{self.base_nb_candles_bearish_buy.value}']* self.low_offset_bearish.value) &
                (last_candle['EWO'] < self.ewo_low.value) &
                (last_candle['volume'] > 0)
                )):
                self.custom_info[pair]['MA_buy'] = 1

        return True


    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        secondlast_candle = dataframe.iloc[-2].squeeze()

        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)
        max_loss = ((trade.open_rate - trade.min_rate) / trade.min_rate)

        if (last_candle is not None):
        
            if (current_rate < last_candle[f'atr_high{self.atr_rolling_length.value}']) & (last_candle['close'] > last_candle[f'atr_high{self.atr_rolling_length.value}']):
            	return 'ATR_stoploss'           
           
            if (current_profit > 0.0) & (self.custom_info[pair]['MA_buy'] == 1): 
            	if ((
		(last_candle[f'pm_{pmaxperiod}_{self.pmaxmultiplier.value}_{self.pmaxlength.value}_{pmaxtype}'] <= last_candle[f'mavalue{self.pmaxlength.value}']) &
                ((last_candle['close'] > last_candle[f'{self.ma_type_index_4.value}_ma_sell_bullish{self.base_nb_candles_bullish_sell.value}']* self.high_offset_bullish.value)) 
            	) or
            	(
		(last_candle[f'pm_{pmaxperiod}_{self.pmaxmultiplier.value}_{self.pmaxlength.value}_{pmaxtype}'] > last_candle[f'mavalue{self.pmaxlength.value}']) &
                ((last_candle['close'] > last_candle[f'{self.ma_type_index_5.value}_ma_sell_bearish{self.base_nb_candles_bearish_sell.value}']* self.high_offset_bearish.value)) 
            	)):
            		self.custom_info[pair]['MA_buy'] = 0
            		return 'offset_sell'
            	if (last_candle['tsi_s'] <= last_candle['tsi_ema_s']) & (secondlast_candle['tsi_s'] > secondlast_candle['tsi_ema_s']) & (current_profit > 0.01):
                    return 'TSI_sell'
                    
            if (current_profit >= 0.005) & (last_candle[f'pm_{pmaxperiod}_{self.pmaxmultiplier.value}_{self.pmaxlength.value}_{pmaxtype}'] > last_candle[f'mavalue{self.pmaxlength.value}']) & (self.custom_info[pair]['MA_buy'] == 1):
                if (last_candle['tsi_f'] <= last_candle['tsi_ema_f']) & (secondlast_candle['tsi_f'] > secondlast_candle['tsi_ema_f']):
                    return 'TSI_dn_sell'
                    
            if (self.custom_info[pair]['MA_buy'] == 1):
            	return None
                    
            if (current_profit > self.sell_custom_profit_11.value) & (last_candle['rsi'] < self.sell_custom_rsi_11.value):
                return 'signal_profit_11'
            if (self.sell_custom_profit_11.value > current_profit > self.sell_custom_profit_10.value) & (last_candle['rsi'] < self.sell_custom_rsi_10.value):
                return 'signal_profit_10'
            if (self.sell_custom_profit_10.value > current_profit > self.sell_custom_profit_9.value) & (last_candle['rsi'] < self.sell_custom_rsi_9.value):
                return 'signal_profit_9'
            if (self.sell_custom_profit_9.value > current_profit > self.sell_custom_profit_8.value) & (last_candle['rsi'] < self.sell_custom_rsi_8.value):
                return 'signal_profit_8'
            if (self.sell_custom_profit_8.value > current_profit > self.sell_custom_profit_7.value) & (last_candle['rsi'] < self.sell_custom_rsi_7.value):
                return 'signal_profit_7'
            if (self.sell_custom_profit_7.value > current_profit > self.sell_custom_profit_6.value) & (last_candle['rsi'] < self.sell_custom_rsi_6.value):
                return 'signal_profit_6'
            if (self.sell_custom_profit_6.value > current_profit > self.sell_custom_profit_5.value) & (last_candle['rsi'] < self.sell_custom_rsi_5.value):
                return 'signal_profit_5'
            elif (self.sell_custom_profit_5.value > current_profit > self.sell_custom_profit_4.value) & (last_candle['rsi'] < self.sell_custom_rsi_4.value):
                return 'signal_profit_4'
            elif (self.sell_custom_profit_4.value > current_profit > self.sell_custom_profit_3.value) & (last_candle['rsi'] < self.sell_custom_rsi_3.value):
                return 'signal_profit_3'
            elif (self.sell_custom_profit_3.value > current_profit > self.sell_custom_profit_2.value) & (last_candle['rsi'] < self.sell_custom_rsi_2.value):
                return 'signal_profit_2'
            elif (self.sell_custom_profit_2.value > current_profit > self.sell_custom_profit_1.value) & (last_candle['rsi'] < self.sell_custom_rsi_1.value):
                return 'signal_profit_1'
            elif (self.sell_custom_profit_1.value > current_profit > self.sell_custom_profit_0.value) & (last_candle['rsi'] < self.sell_custom_rsi_0.value):
                return 'signal_profit_0'

            # check if close is under EMA200
            elif (current_profit > self.sell_custom_under_profit_11.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_11.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_11'
            elif (self.sell_custom_under_profit_11.value > current_profit > self.sell_custom_under_profit_10.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_10.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_10'
            elif (self.sell_custom_under_profit_10.value > current_profit > self.sell_custom_under_profit_9.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_9.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_9'
            elif (self.sell_custom_under_profit_9.value > current_profit > self.sell_custom_under_profit_8.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_8.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_8'
            elif (self.sell_custom_under_profit_8.value > current_profit > self.sell_custom_under_profit_7.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_7.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_7'
            elif (self.sell_custom_under_profit_7.value > current_profit > self.sell_custom_under_profit_6.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_6.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_6'
            elif (self.sell_custom_under_profit_6.value > current_profit > self.sell_custom_under_profit_5.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_5.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_5'
            elif (self.sell_custom_under_profit_5.value > current_profit > self.sell_custom_under_profit_4.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_4.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_4'
            elif (self.sell_custom_under_profit_4.value > current_profit > self.sell_custom_under_profit_3.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_3.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_3'
            elif (self.sell_custom_under_profit_3.value > current_profit > self.sell_custom_under_profit_2.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_2.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_2'
            elif (self.sell_custom_under_profit_2.value > current_profit > self.sell_custom_under_profit_1.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_1.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_1'
            elif (self.sell_custom_under_profit_1.value > current_profit > self.sell_custom_under_profit_0.value) & (last_candle['rsi'] < self.sell_custom_under_rsi_0.value) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_u_0'

            # check if the pair is "pumped"

            elif (last_candle['sell_pump_48_1_1h']) & (current_profit > self.sell_custom_pump_profit_1_5.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_1_5.value):
                return 'signal_profit_p_1_5'
            elif (last_candle['sell_pump_48_1_1h']) & (self.sell_custom_pump_profit_1_5.value > current_profit > self.sell_custom_pump_profit_1_4.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_1_4.value):
                return 'signal_profit_p_1_4'
            elif (last_candle['sell_pump_48_1_1h']) & (self.sell_custom_pump_profit_1_4.value > current_profit > self.sell_custom_pump_profit_1_3.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_1_3.value):
                return 'signal_profit_p_1_3'
            elif (last_candle['sell_pump_48_1_1h']) & (self.sell_custom_pump_profit_1_3.value > current_profit > self.sell_custom_pump_profit_1_2.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_1_2.value):
                return 'signal_profit_p_1_2'
            elif (last_candle['sell_pump_48_1_1h']) & (self.sell_custom_pump_profit_1_2.value > current_profit > self.sell_custom_pump_profit_1_1.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_1_1.value):
                return 'signal_profit_p_1_1'

            elif (last_candle['sell_pump_36_1_1h']) & (current_profit > self.sell_custom_pump_profit_2_5.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_2_5.value):
                return 'signal_profit_p_2_5'
            elif (last_candle['sell_pump_36_1_1h']) & (self.sell_custom_pump_profit_2_5.value > current_profit > self.sell_custom_pump_profit_2_4.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_2_4.value):
                return 'signal_profit_p_2_4'
            elif (last_candle['sell_pump_36_1_1h']) & (self.sell_custom_pump_profit_2_4.value > current_profit > self.sell_custom_pump_profit_2_3.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_2_3.value):
                return 'signal_profit_p_2_3'
            elif (last_candle['sell_pump_36_1_1h']) & (self.sell_custom_pump_profit_2_3.value > current_profit > self.sell_custom_pump_profit_2_2.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_2_2.value):
                return 'signal_profit_p_2_2'
            elif (last_candle['sell_pump_36_1_1h']) & (self.sell_custom_pump_profit_2_2.value > current_profit > self.sell_custom_pump_profit_2_1.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_2_1.value):
                return 'signal_profit_p_2_1'

            elif (last_candle['sell_pump_24_1_1h']) & (current_profit > self.sell_custom_pump_profit_3_5.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_3_5.value):
                return 'signal_profit_p_3_5'
            elif (last_candle['sell_pump_24_1_1h']) & (self.sell_custom_pump_profit_3_5.value > current_profit > self.sell_custom_pump_profit_3_4.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_3_4.value):
                return 'signal_profit_p_3_4'
            elif (last_candle['sell_pump_24_1_1h']) & (self.sell_custom_pump_profit_3_4.value > current_profit > self.sell_custom_pump_profit_3_3.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_3_3.value):
                return 'signal_profit_p_3_3'
            elif (last_candle['sell_pump_24_1_1h']) & (self.sell_custom_pump_profit_3_3.value > current_profit > self.sell_custom_pump_profit_3_2.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_3_2.value):
                return 'signal_profit_p_3_2'
            elif (last_candle['sell_pump_24_1_1h']) & (self.sell_custom_pump_profit_3_2.value > current_profit > self.sell_custom_pump_profit_3_1.value) & (last_candle['rsi'] < self.sell_custom_pump_rsi_3_1.value):
                return 'signal_profit_p_3_1'

            elif (self.sell_custom_dec_profit_max_1.value > current_profit > self.sell_custom_dec_profit_min_1.value) & (last_candle['sma_200_dec_20']):
                return 'signal_profit_d_1'
            elif (self.sell_custom_dec_profit_max_2.value > current_profit > self.sell_custom_dec_profit_min_2.value) & (last_candle['close'] < last_candle['ema_100']):
                return 'signal_profit_d_2'

            # Trailing
            elif (self.sell_trail_profit_max_1.value > current_profit > self.sell_trail_profit_min_1.value) & (self.sell_trail_rsi_min_1.value < last_candle['rsi'] < self.sell_trail_rsi_max_1.value) & (max_profit > (current_profit + self.sell_trail_down_1.value)):
                return 'signal_profit_t_1'
            elif (self.sell_trail_profit_max_2.value > current_profit > self.sell_trail_profit_min_2.value) & (self.sell_trail_rsi_min_2.value < last_candle['rsi'] < self.sell_trail_rsi_max_2.value) & (max_profit > (current_profit + self.sell_trail_down_2.value)):
                return 'signal_profit_t_2'
            elif (self.sell_trail_profit_max_3.value > current_profit > self.sell_trail_profit_min_3.value) & (max_profit > (current_profit + self.sell_trail_down_3.value)) & (last_candle['sma_200_dec_20_1h']):
                return 'signal_profit_t_3'

            elif (last_candle['close'] < last_candle['ema_200']) & (current_profit > self.sell_trail_profit_min_3.value) & (current_profit < self.sell_trail_profit_max_3.value) & (max_profit > (current_profit + self.sell_trail_down_3.value)):
                return 'signal_profit_u_t_1'

            # elif (last_candle['sell_pump_48_1_1h']) & (0.06 > current_profit > 0.04) & (last_candle['rsi'] < 54.0) & (current_time - timedelta(minutes=30) < trade.open_date_utc):
            #     return 'signal_profit_p_s_1'

            elif (current_profit > 0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_profit_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_profit_under_rsi_diff_1.value):
                return 'signal_profit_u_e_1'

            elif (current_profit < -0.0) & (last_candle['close'] < last_candle['ema_200']) & (((last_candle['ema_200'] - last_candle['close']) / last_candle['close']) < self.sell_custom_stoploss_under_rel_1.value) & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_under_rsi_diff_1.value) & (last_candle['sma_200_dec_24']) & (current_time - timedelta(minutes=720) > trade.open_date_utc):
                return 'signal_stoploss_u_1'

            elif (self.sell_custom_stoploss_long_profit_min_1.value < current_profit < self.sell_custom_stoploss_long_profit_max_1.value) & (current_profit > (-max_loss + self.sell_custom_stoploss_long_recover_1.value)) & (last_candle['close'] < last_candle['ema_200'])  & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_long_rsi_diff_1.value) & (last_candle['sma_200_dec_24']) & (current_time - timedelta(minutes=1200) > trade.open_date_utc):
                return 'signal_stoploss_l_r_u_1'

            elif (current_profit < -0.0) & (current_profit > (-max_loss + self.sell_custom_stoploss_long_recover_2.value)) & (last_candle['close'] < last_candle['ema_200'])  & (last_candle['rsi'] > last_candle['rsi_1h'] + self.sell_custom_stoploss_long_rsi_diff_2.value) & (last_candle['sma_200_dec_24']) & (current_time - timedelta(minutes=1200) > trade.open_date_utc):
                return 'signal_stoploss_l_r_u_2'

            elif (self.sell_custom_pump_dec_profit_max_1.value > current_profit > self.sell_custom_pump_dec_profit_min_1.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['sma_200_dec_20']) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_p_d_1'
            elif (self.sell_custom_pump_dec_profit_max_2.value > current_profit > self.sell_custom_pump_dec_profit_min_2.value) & (last_candle['sell_pump_48_2_1h']) & (last_candle['sma_200_dec_20']) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_p_d_2'
            elif (self.sell_custom_pump_dec_profit_max_3.value > current_profit > self.sell_custom_pump_dec_profit_min_3.value) & (last_candle['sell_pump_48_3_1h']) & (last_candle['sma_200_dec_20']) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_p_d_3'
            elif (self.sell_custom_pump_dec_profit_max_4.value > current_profit > self.sell_custom_pump_dec_profit_min_4.value) & (last_candle['sma_200_dec_20']) & (last_candle['sell_pump_24_2_1h']):
                return 'signal_profit_p_d_4'

            # Pumped 48h 1, under EMA200
            elif (self.sell_custom_pump_under_profit_max_1.value > current_profit > self.sell_custom_pump_under_profit_min_1.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['close'] < last_candle['ema_200']):
                return 'signal_profit_p_u_1'

            # Pumped 36h 2, trail 1
            elif (last_candle['sell_pump_36_2_1h']) & (self.sell_custom_pump_trail_profit_max_1.value > current_profit > self.sell_custom_pump_trail_profit_min_1.value) & (self.sell_custom_pump_trail_rsi_min_1.value < last_candle['rsi'] < self.sell_custom_pump_trail_rsi_max_1.value) & (max_profit > (current_profit + self.sell_custom_pump_trail_down_1.value)):
                return 'signal_profit_p_t_1'

            # elif (max_profit < self.sell_custom_stoploss_pump_max_profit_1.value) & (self.sell_custom_stoploss_pump_min_1.value < current_profit < self.sell_custom_stoploss_pump_max_1.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['sma_200_dec_20']) & (last_candle['close'] < (last_candle['ema_200'] * self.sell_custom_stoploss_pump_ma_offset_1.value)):
            #     return 'signal_stoploss_p_1'

            elif (max_profit < self.sell_custom_stoploss_pump_max_profit_2.value) & (current_profit < self.sell_custom_stoploss_pump_loss_2.value) & (last_candle['sell_pump_48_1_1h']) & (last_candle['sma_200_dec_20_1h']) & (last_candle['close'] < (last_candle['ema_200'] * self.sell_custom_stoploss_pump_ma_offset_2.value)):
                return 'signal_stoploss_p_2'

            elif (max_profit < self.sell_custom_stoploss_pump_max_profit_3.value) & (current_profit < self.sell_custom_stoploss_pump_loss_3.value) & (last_candle['sell_pump_36_3_1h']) & (last_candle['close'] < (last_candle['ema_200'] * self.sell_custom_stoploss_pump_ma_offset_3.value)):
                return 'signal_stoploss_p_3'

            # Recover
            elif (max_loss > self.sell_custom_recover_min_loss_1.value) & (current_profit > self.sell_custom_recover_profit_1.value):
                return 'signal_profit_r_1'

            elif (max_loss > self.sell_custom_recover_min_loss_2.value) & (self.sell_custom_recover_profit_max_2.value > current_profit > self.sell_custom_recover_profit_min_2.value) & (last_candle['rsi'] < self.sell_custom_recover_rsi_2.value):
                return 'signal_profit_r_2'

            # Take profit for long duration trades
            elif (self.sell_custom_long_profit_min_1.value < current_profit < self.sell_custom_long_profit_max_1.value) & (current_time - timedelta(minutes=self.sell_custom_long_duration_min_1.value) > trade.open_date_utc):
                return 'signal_profit_l_1'

        return None

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, sell_reason: str, **kwargs) -> bool:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        # Prevent sell, if there is more potential, in order to maximize profit
        if (last_candle is not None):
            if (sell_reason == 'sell_signal') & (self.custom_info[pair]['MA_buy'] == 1):
            	return False
        return True


    def range_percent_change(self, dataframe: DataFrame, method, length: int) -> float:
        """
        Rolling Percentage Change Maximum across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param method: High to Low / Open to Close
        :param length: int The length to look back
        """
        df = dataframe.copy()
        if method == 'HL':
            return ((df['high'].rolling(length).max() - df['low'].rolling(length).min()) / df['low'].rolling(length).min())
        elif method == 'OC':
            return ((df['open'].rolling(length).max() - df['close'].rolling(length).min()) / df['close'].rolling(length).min())
        else:
            raise ValueError(f"Method {method} not defined!")

    def top_percent_change(self, dataframe: DataFrame, length: int) -> float:
        """
        Percentage change of the current close from the range maximum Open price

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        df = dataframe.copy()
        if length == 0:
            return ((df['open'] - df['close']) / df['close'])
        else:
            return ((df['open'].rolling(length).max() - df['close']) / df['close'])

    def range_maxgap(self, dataframe: DataFrame, length: int) -> float:
        """
        Maximum Price Gap across interval.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        df = dataframe.copy()
        return (df['open'].rolling(length).max() - df['close'].rolling(length).min())

    def range_maxgap_adjusted(self, dataframe: DataFrame, length: int, adjustment: float) -> float:
        """
        Maximum Price Gap across interval adjusted.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        :param adjustment: int The adjustment to be applied
        """
        return (self.range_maxgap(dataframe,length) / adjustment)

    def range_height(self, dataframe: DataFrame, length: int) -> float:
        """
        Current close distance to range bottom.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        """
        df = dataframe.copy()
        return (df['close'] - df['close'].rolling(length).min())

    def safe_pump(self, dataframe: DataFrame, length: int, thresh: float, pull_thresh: float) -> bool:
        """
        Determine if entry after a pump is safe.

        :param dataframe: DataFrame The original OHLC dataframe
        :param length: int The length to look back
        :param thresh: int Maximum percentage change threshold
        :param pull_thresh: int Pullback from interval maximum threshold
        """
        df = dataframe.copy()
        return (df[f'oc_pct_change_{length}'] < thresh) | (self.range_maxgap_adjusted(df, length, pull_thresh) > self.range_height(df, length))

    def safe_dips(self, dataframe: DataFrame, thresh_0, thresh_2, thresh_12, thresh_144) -> bool:
        """
        Determine if dip is safe to enter.

        :param dataframe: DataFrame The original OHLC dataframe
        :param thresh_0: Threshold value for 0 length top pct change
        :param thresh_2: Threshold value for 2 length top pct change
        :param thresh_12: Threshold value for 12 length top pct change
        :param thresh_144: Threshold value for 144 length top pct change
        """
        return ((dataframe['tpct_change_0'] < thresh_0) &
                (dataframe['tpct_change_2'] < thresh_2) &
                (dataframe['tpct_change_12'] < thresh_12) &
                (dataframe['tpct_change_144'] < thresh_144))

    def informative_pairs(self):
        # get access to all pairs available in whitelist.
        pairs = self.dp.current_whitelist()
        # Assign tf to each pair so they can be downloaded and cached for strategy.
        informative_pairs = [(pair, '1h') for pair in pairs]
        return informative_pairs

    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)

        # EMA
        informative_1h['ema_12'] = ta.EMA(informative_1h, timeperiod=12)
        informative_1h['ema_15'] = ta.EMA(informative_1h, timeperiod=15)
        informative_1h['ema_20'] = ta.EMA(informative_1h, timeperiod=20)
        informative_1h['ema_26'] = ta.EMA(informative_1h, timeperiod=26)
        informative_1h['ema_35'] = ta.EMA(informative_1h, timeperiod=35)
        informative_1h['ema_50'] = ta.EMA(informative_1h, timeperiod=50)
        informative_1h['ema_100'] = ta.EMA(informative_1h, timeperiod=100)
        informative_1h['ema_200'] = ta.EMA(informative_1h, timeperiod=200)

        # SMA
        informative_1h['sma_200'] = ta.SMA(informative_1h, timeperiod=200)
        informative_1h['sma_200_dec_20'] = informative_1h['sma_200'] < informative_1h['sma_200'].shift(20)

        # RSI
        informative_1h['rsi'] = ta.RSI(informative_1h, timeperiod=14)

        # BB
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative_1h), window=20, stds=2)
        informative_1h['bb20_2_low'] = bollinger['lower']
        informative_1h['bb20_2_mid'] = bollinger['mid']
        informative_1h['bb20_2_upp'] = bollinger['upper']

        # Chaikin Money Flow
        informative_1h['cmf'] = chaikin_money_flow(informative_1h, 20)

        # Pump protections
        informative_1h['hl_pct_change_48'] = self.range_percent_change(informative_1h, 'HL', 48)
        informative_1h['hl_pct_change_36'] = self.range_percent_change(informative_1h, 'HL', 36)
        informative_1h['hl_pct_change_24'] = self.range_percent_change(informative_1h, 'HL', 24)

        informative_1h['oc_pct_change_48'] = self.range_percent_change(informative_1h, 'OC', 48)
        informative_1h['oc_pct_change_36'] = self.range_percent_change(informative_1h, 'OC', 36)
        informative_1h['oc_pct_change_24'] = self.range_percent_change(informative_1h, 'OC', 24)

        informative_1h['safe_pump_24_10'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_10_24.value, self.buy_pump_pull_threshold_10_24.value)
        informative_1h['safe_pump_36_10'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_10_36.value, self.buy_pump_pull_threshold_10_36.value)
        informative_1h['safe_pump_48_10'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_10_48.value, self.buy_pump_pull_threshold_10_48.value)

        informative_1h['safe_pump_24_20'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_20_24.value, self.buy_pump_pull_threshold_20_24.value)
        informative_1h['safe_pump_36_20'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_20_36.value, self.buy_pump_pull_threshold_20_36.value)
        informative_1h['safe_pump_48_20'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_20_48.value, self.buy_pump_pull_threshold_20_48.value)

        informative_1h['safe_pump_24_30'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_30_24.value, self.buy_pump_pull_threshold_30_24.value)
        informative_1h['safe_pump_36_30'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_30_36.value, self.buy_pump_pull_threshold_30_36.value)
        informative_1h['safe_pump_48_30'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_30_48.value, self.buy_pump_pull_threshold_30_48.value)

        informative_1h['safe_pump_24_40'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_40_24.value, self.buy_pump_pull_threshold_40_24.value)
        informative_1h['safe_pump_36_40'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_40_36.value, self.buy_pump_pull_threshold_40_36.value)
        informative_1h['safe_pump_48_40'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_40_48.value, self.buy_pump_pull_threshold_40_48.value)

        informative_1h['safe_pump_24_50'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_50_24.value, self.buy_pump_pull_threshold_50_24.value)
        informative_1h['safe_pump_36_50'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_50_36.value, self.buy_pump_pull_threshold_50_36.value)
        informative_1h['safe_pump_48_50'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_50_48.value, self.buy_pump_pull_threshold_50_48.value)

        informative_1h['safe_pump_24_60'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_60_24.value, self.buy_pump_pull_threshold_60_24.value)
        informative_1h['safe_pump_36_60'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_60_36.value, self.buy_pump_pull_threshold_60_36.value)
        informative_1h['safe_pump_48_60'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_60_48.value, self.buy_pump_pull_threshold_60_48.value)

        informative_1h['safe_pump_24_70'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_70_24.value, self.buy_pump_pull_threshold_70_24.value)
        informative_1h['safe_pump_36_70'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_70_36.value, self.buy_pump_pull_threshold_70_36.value)
        informative_1h['safe_pump_48_70'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_70_48.value, self.buy_pump_pull_threshold_70_48.value)

        informative_1h['safe_pump_24_80'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_80_24.value, self.buy_pump_pull_threshold_80_24.value)
        informative_1h['safe_pump_36_80'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_80_36.value, self.buy_pump_pull_threshold_80_36.value)
        informative_1h['safe_pump_48_80'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_80_48.value, self.buy_pump_pull_threshold_80_48.value)

        informative_1h['safe_pump_24_90'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_90_24.value, self.buy_pump_pull_threshold_90_24.value)
        informative_1h['safe_pump_36_90'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_90_36.value, self.buy_pump_pull_threshold_90_36.value)
        informative_1h['safe_pump_48_90'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_90_48.value, self.buy_pump_pull_threshold_90_48.value)

        informative_1h['safe_pump_24_100'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_100_24.value, self.buy_pump_pull_threshold_100_24.value)
        informative_1h['safe_pump_36_100'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_100_36.value, self.buy_pump_pull_threshold_100_36.value)
        informative_1h['safe_pump_48_100'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_100_48.value, self.buy_pump_pull_threshold_100_48.value)

        informative_1h['safe_pump_24_110'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_110_24.value, self.buy_pump_pull_threshold_110_24.value)
        informative_1h['safe_pump_36_110'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_110_36.value, self.buy_pump_pull_threshold_110_36.value)
        informative_1h['safe_pump_48_110'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_110_48.value, self.buy_pump_pull_threshold_110_48.value)

        informative_1h['safe_pump_24_120'] = self.safe_pump(informative_1h, 24, self.buy_pump_threshold_120_24.value, self.buy_pump_pull_threshold_120_24.value)
        informative_1h['safe_pump_36_120'] = self.safe_pump(informative_1h, 36, self.buy_pump_threshold_120_36.value, self.buy_pump_pull_threshold_120_36.value)
        informative_1h['safe_pump_48_120'] = self.safe_pump(informative_1h, 48, self.buy_pump_threshold_120_48.value, self.buy_pump_pull_threshold_120_48.value)

        informative_1h['sell_pump_48_1'] = (informative_1h['hl_pct_change_48'] > self.sell_pump_threshold_48_1.value)
        informative_1h['sell_pump_48_2'] = (informative_1h['hl_pct_change_48'] > self.sell_pump_threshold_48_2.value)
        informative_1h['sell_pump_48_3'] = (informative_1h['hl_pct_change_48'] > self.sell_pump_threshold_48_3.value)

        informative_1h['sell_pump_36_1'] = (informative_1h['hl_pct_change_36'] > self.sell_pump_threshold_36_1.value)
        informative_1h['sell_pump_36_2'] = (informative_1h['hl_pct_change_36'] > self.sell_pump_threshold_36_2.value)
        informative_1h['sell_pump_36_3'] = (informative_1h['hl_pct_change_36'] > self.sell_pump_threshold_36_3.value)

        informative_1h['sell_pump_24_1'] = (informative_1h['hl_pct_change_24'] > self.sell_pump_threshold_24_1.value)
        informative_1h['sell_pump_24_2'] = (informative_1h['hl_pct_change_24'] > self.sell_pump_threshold_24_2.value)
        informative_1h['sell_pump_24_3'] = (informative_1h['hl_pct_change_24'] > self.sell_pump_threshold_24_3.value)

        return informative_1h

    def normal_tf_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
	#PMAX
        heikinashi = qtpylib.heikinashi(dataframe)
        heikinashi["volume"] = dataframe["volume"]
        
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close'])/4

        for mult in self.pmaxmultiplier.range:
        	for leng in self.pmaxlength.range:
        		pmdf = PMAX(heikinashi, MAtype=pmaxtype, length=leng, multiplier=mult, period=pmaxperiod, src=3)
        		dataframe[f'pm_{pmaxperiod}_{mult}_{leng}_{pmaxtype}'] = pmdf[f'pm_{pmaxperiod}_{mult}_{leng}_{pmaxtype}']
        		dataframe[f'mavalue{leng}'] = ta.EMA(dataframe['source'], timeperiod=leng)

        # BB 40 - STD2
        bb_40_std2 = qtpylib.bollinger_bands(dataframe['close'], window=40, stds=2)
        dataframe['bb40_2_low']= bb_40_std2['lower']
        dataframe['bb40_2_mid'] = bb_40_std2['mid']
        dataframe['bb40_2_delta'] = (bb_40_std2['mid'] - dataframe['bb40_2_low']).abs()
        dataframe['closedelta'] = (dataframe['close'] - dataframe['close'].shift()).abs()
        dataframe['tail'] = (dataframe['close'] - dataframe['bb40_2_low']).abs()

        # BB 20 - STD2
        bb_20_std2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb20_2_low'] = bb_20_std2['lower']
        dataframe['bb20_2_mid'] = bb_20_std2['mid']
        dataframe['bb20_2_upp'] = bb_20_std2['upper']

        # EMA 200
        dataframe['ema_12'] = ta.EMA(dataframe, timeperiod=12)
        dataframe['ema_15'] = ta.EMA(dataframe, timeperiod=15)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_26'] = ta.EMA(dataframe, timeperiod=26)
        dataframe['ema_35'] = ta.EMA(dataframe, timeperiod=35)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)

        # SMA
        dataframe['sma_5'] = ta.SMA(dataframe, timeperiod=5)
        dataframe['sma_30'] = ta.SMA(dataframe, timeperiod=30)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        dataframe['sma_200_dec_20'] = dataframe['sma_200'] < dataframe['sma_200'].shift(20)
        dataframe['sma_200_dec_24'] = dataframe['sma_200'] < dataframe['sma_200'].shift(24)

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # EWO
        dataframe['ewo'] = EWO(dataframe, 50, 200)

        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Chopiness
        dataframe['chop']= qtpylib.chopiness(dataframe, 14)

        # Zero-Lag EMA
        dataframe['zema'] = zema(dataframe, period=61)

        # Dip protection
        dataframe['tpct_change_0']   = self.top_percent_change(dataframe,0)
        dataframe['tpct_change_2']   = self.top_percent_change(dataframe,2)
        dataframe['tpct_change_12']  = self.top_percent_change(dataframe,12)
        dataframe['tpct_change_144'] = self.top_percent_change(dataframe,144)

        dataframe['safe_dips_10']  = self.safe_dips(dataframe, self.buy_dip_threshold_10_1.value, self.buy_dip_threshold_10_2.value, self.buy_dip_threshold_10_3.value, self.buy_dip_threshold_10_4.value)
        dataframe['safe_dips_20']  = self.safe_dips(dataframe, self.buy_dip_threshold_20_1.value, self.buy_dip_threshold_20_2.value, self.buy_dip_threshold_20_3.value, self.buy_dip_threshold_20_4.value)
        dataframe['safe_dips_30']  = self.safe_dips(dataframe, self.buy_dip_threshold_30_1.value, self.buy_dip_threshold_30_2.value, self.buy_dip_threshold_30_3.value, self.buy_dip_threshold_30_4.value)
        dataframe['safe_dips_40']  = self.safe_dips(dataframe, self.buy_dip_threshold_40_1.value, self.buy_dip_threshold_40_2.value, self.buy_dip_threshold_40_3.value, self.buy_dip_threshold_40_4.value)
        dataframe['safe_dips_50']  = self.safe_dips(dataframe, self.buy_dip_threshold_50_1.value, self.buy_dip_threshold_50_2.value, self.buy_dip_threshold_50_3.value, self.buy_dip_threshold_50_4.value)
        dataframe['safe_dips_60']  = self.safe_dips(dataframe, self.buy_dip_threshold_60_1.value, self.buy_dip_threshold_60_2.value, self.buy_dip_threshold_60_3.value, self.buy_dip_threshold_60_4.value)
        dataframe['safe_dips_70']  = self.safe_dips(dataframe, self.buy_dip_threshold_70_1.value, self.buy_dip_threshold_70_2.value, self.buy_dip_threshold_70_3.value, self.buy_dip_threshold_70_4.value)
        dataframe['safe_dips_80']  = self.safe_dips(dataframe, self.buy_dip_threshold_80_1.value, self.buy_dip_threshold_80_2.value, self.buy_dip_threshold_80_3.value, self.buy_dip_threshold_80_4.value)
        dataframe['safe_dips_90']  = self.safe_dips(dataframe, self.buy_dip_threshold_90_1.value, self.buy_dip_threshold_90_2.value, self.buy_dip_threshold_90_3.value, self.buy_dip_threshold_90_4.value)
        dataframe['safe_dips_100'] = self.safe_dips(dataframe, self.buy_dip_threshold_100_1.value, self.buy_dip_threshold_100_2.value, self.buy_dip_threshold_100_3.value, self.buy_dip_threshold_100_4.value)
        dataframe['safe_dips_110'] = self.safe_dips(dataframe, self.buy_dip_threshold_110_1.value, self.buy_dip_threshold_110_2.value, self.buy_dip_threshold_110_3.value, self.buy_dip_threshold_110_4.value)

        # Volume
        dataframe['volume_mean_4'] = dataframe['volume'].rolling(4).mean().shift(1)
        dataframe['volume_mean_30'] = dataframe['volume'].rolling(30).mean()
        
        for mval in self.ma_type_index_1.range:
            for val in self.base_nb_candles_bullish_buy.range:
                dataframe[f'{mval}_ma_buy_bullish{val}'] = ma_types[list(ma_types)[mval]](dataframe, timeperiod=val)

        for mval in self.ma_type_index_2.range:
            for val in self.base_nb_candles_bearish_buy.range:
                dataframe[f'{mval}_ma_buy_bearish{val}'] = ma_types[list(ma_types)[mval]](dataframe, timeperiod=val)
                
        for mval in self.ma_type_index_4.range:
            for val in self.base_nb_candles_bullish_sell.range:
                dataframe[f'{mval}_ma_sell_bullish{val}'] = ma_types[list(ma_types)[mval]](dataframe, timeperiod=val)

        for mval in self.ma_type_index_5.range:
            for val in self.base_nb_candles_bearish_sell.range:
                dataframe[f'{mval}_ma_sell_bearish{val}'] = ma_types[list(ma_types)[mval]](dataframe, timeperiod=val)

        dataframe['EWO'] = EWO(dataframe, 50, 200)

        datas = TSIIndicator(dataframe['close'], window_slow=20, window_fast=5)
        dataframe['tsi_s'] = datas.tsi()
        dataframe['tsi_ema_s'] = ta.EMA(dataframe['tsi_s'], timeperiod=5)
        
        dataf = TSIIndicator(dataframe['close'], window_slow=4, window_fast=2)
        dataframe['tsi_f'] = dataf.tsi()
        dataframe['tsi_ema_f'] = ta.EMA(dataframe['tsi_f'], timeperiod=2)
        
#        dataframe['adx'] = ta.ADX(dataframe)

        # RSI
        dataframe['rsi14'] = ta.RSI(dataframe, timeperiod=14)
        
        #ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        for i in self.atr_rolling_length.range:
        	dataframe[f'atr_high{i}'] = (dataframe['high'] - (dataframe['atr'] * self.atr_multiplier.value)).rolling(i).max()
        dataframe['ema_atr'] = ta.EMA(dataframe['atr'], timeperiod=14)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # The indicators for the 1h informative timeframe
        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)

        # The indicators for the normal (5m) timeframe
        dataframe = self.normal_tf_indicators(dataframe, metadata)

        return dataframe


    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        
        conditions.append(
            (
		        (dataframe[f'pm_{pmaxperiod}_{self.pmaxmultiplier.value}_{self.pmaxlength.value}_{pmaxtype}'] <= dataframe[f'mavalue{self.pmaxlength.value}']) &
                (dataframe['close'] < dataframe[f'{self.ma_type_index_1.value}_ma_buy_bullish{self.base_nb_candles_bullish_buy.value}']* self.low_offset_bullish.value) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi14'] < self.rsi_buy_bullish.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
		        (dataframe[f'pm_{pmaxperiod}_{self.pmaxmultiplier.value}_{self.pmaxlength.value}_{pmaxtype}'] <= dataframe[f'mavalue{self.pmaxlength.value}']) &
                (dataframe['close'] < dataframe[f'{self.ma_type_index_1.value}_ma_buy_bullish{self.base_nb_candles_bullish_buy.value}']* self.low_offset_bullish.value) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
		        (dataframe[f'pm_{pmaxperiod}_{self.pmaxmultiplier.value}_{self.pmaxlength.value}_{pmaxtype}'] > dataframe[f'mavalue{self.pmaxlength.value}']) &
                (dataframe['close'] < dataframe[f'{self.ma_type_index_2.value}_ma_buy_bearish{self.base_nb_candles_bearish_buy.value}']* self.low_offset_bearish.value) &
                (dataframe['EWO'] > self.ewo_high.value) &
                (dataframe['rsi14'] < self.rsi_buy_bearish.value) &

                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
		        (dataframe[f'pm_{pmaxperiod}_{self.pmaxmultiplier.value}_{self.pmaxlength.value}_{pmaxtype}'] > dataframe[f'mavalue{self.pmaxlength.value}']) &
                (dataframe['close'] < dataframe[f'{self.ma_type_index_2.value}_ma_buy_bearish{self.base_nb_candles_bearish_buy.value}']* self.low_offset_bearish.value) &
                (dataframe['EWO'] < self.ewo_low.value) &
                (dataframe['volume'] > 0)
            )
        )        
        
        
        # Protections
        buy_01_protections = [True]
        if self.buy_01_protection__ema_fast.value:
            buy_01_protections.append(dataframe[f"ema_{self.buy_01_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_01_protection__ema_slow.value:
            buy_01_protections.append(dataframe[f"ema_{self.buy_01_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_01_protection__close_above_ema_fast.value:
            buy_01_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_01_protection__close_above_ema_fast_len.value}"])
        if self.buy_01_protection__close_above_ema_slow.value:
            buy_01_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_01_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_01_protection__sma200_rising.value:
            buy_01_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_01_protection__sma200_rising_val.value)))
        if self.buy_01_protection__sma200_1h_rising.value:
            buy_01_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_01_protection__sma200_1h_rising_val.value)))
        if self.buy_01_protection__safe_dips.value:
            buy_01_protections.append(dataframe[f"safe_dips_{self.buy_01_protection__safe_dips_type.value}"])
        if self.buy_01_protection__safe_pump.value:
            buy_01_protections.append(dataframe[f"safe_pump_{self.buy_01_protection__safe_pump_period.value}_{self.buy_01_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_01_logic = []
        buy_01_logic.append(reduce(lambda x, y: x & y, buy_01_protections))
        buy_01_logic.append(((dataframe['close'] - dataframe['open'].rolling(36).min()) / dataframe['open'].rolling(36).min()) > self.buy_min_inc_1.value)
        buy_01_logic.append(dataframe['rsi_1h'] > self.buy_rsi_1h_min_1.value)
        buy_01_logic.append(dataframe['rsi_1h'] < self.buy_rsi_1h_max_1.value)
        buy_01_logic.append(dataframe['rsi'] < self.buy_rsi_1.value)
        buy_01_logic.append(dataframe['mfi'] < self.buy_mfi_1.value)
        buy_01_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_01_trigger'] = reduce(lambda x, y: x & y, buy_01_logic)
        if self.buy_condition_1_enable.value:
            conditions.append(dataframe['buy_01_trigger'])

        # Protections
        buy_02_protections = [True]
        if self.buy_02_protection__ema_fast.value:
            buy_02_protections.append(dataframe[f"ema_{self.buy_02_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_02_protection__ema_slow.value:
            buy_02_protections.append(dataframe[f"ema_{self.buy_02_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_02_protection__close_above_ema_fast.value:
            buy_02_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_02_protection__close_above_ema_fast_len.value}"])
        if self.buy_02_protection__close_above_ema_slow.value:
            buy_02_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_02_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_02_protection__sma200_rising.value:
            buy_02_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_02_protection__sma200_rising_val.value)))
        if self.buy_02_protection__sma200_1h_rising.value:
            buy_02_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_02_protection__sma200_1h_rising_val.value)))
        if self.buy_02_protection__safe_dips.value:
            buy_02_protections.append(dataframe[f"safe_dips_{self.buy_02_protection__safe_dips_type.value}"])
        if self.buy_02_protection__safe_pump.value:
            buy_02_protections.append(dataframe[f"safe_pump_{self.buy_02_protection__safe_pump_period.value}_{self.buy_02_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_02_logic = []
        buy_02_logic.append(reduce(lambda x, y: x & y, buy_02_protections))
        #buy_02_logic.append(dataframe['volume_mean_4'] * self.buy_volume_2.value > dataframe['volume'])
        buy_02_logic.append(dataframe['rsi'] < dataframe['rsi_1h'] - self.buy_rsi_1h_diff_2.value)
        buy_02_logic.append(dataframe['mfi'] < self.buy_mfi_2.value)
        buy_02_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_2.value))
        buy_02_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_02_trigger'] = reduce(lambda x, y: x & y, buy_02_logic)
        if self.buy_condition_2_enable.value:
            conditions.append(dataframe['buy_02_trigger'])

        # Protections
        buy_03_protections = [True]
        if self.buy_03_protection__ema_fast.value:
            buy_03_protections.append(dataframe[f"ema_{self.buy_03_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_03_protection__ema_slow.value:
            buy_03_protections.append(dataframe[f"ema_{self.buy_03_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_03_protection__close_above_ema_fast.value:
            buy_03_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_03_protection__close_above_ema_fast_len.value}"])
        if self.buy_03_protection__close_above_ema_slow.value:
            buy_03_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_03_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_03_protection__sma200_rising.value:
            buy_03_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_03_protection__sma200_rising_val.value)))
        if self.buy_03_protection__sma200_1h_rising.value:
            buy_03_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_03_protection__sma200_1h_rising_val.value)))
        if self.buy_03_protection__safe_dips.value:
            buy_03_protections.append(dataframe[f"safe_dips_{self.buy_03_protection__safe_dips_type.value}"])
        if self.buy_03_protection__safe_pump.value:
            buy_03_protections.append(dataframe[f"safe_pump_{self.buy_03_protection__safe_pump_period.value}_{self.buy_03_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)
        buy_03_protections.append(dataframe['close'] > (dataframe['ema_200_1h'] * self.buy_ema_rel_3.value))

        # Logic
        buy_03_logic = []
        buy_03_logic.append(reduce(lambda x, y: x & y, buy_03_protections))
        buy_03_logic.append(dataframe['bb40_2_low'].shift().gt(0))
        buy_03_logic.append(dataframe['bb40_2_delta'].gt(dataframe['close'] * self.buy_bb40_bbdelta_close_3.value))
        buy_03_logic.append(dataframe['closedelta'].gt(dataframe['close'] * self.buy_bb40_closedelta_close_3.value))
        buy_03_logic.append(dataframe['tail'].lt(dataframe['bb40_2_delta'] * self.buy_bb40_tail_bbdelta_3.value))
        buy_03_logic.append(dataframe['close'].lt(dataframe['bb40_2_low'].shift()))
        buy_03_logic.append(dataframe['close'].le(dataframe['close'].shift()))
        buy_03_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_03_trigger'] = reduce(lambda x, y: x & y, buy_03_logic)
        if self.buy_condition_3_enable.value:
            conditions.append(dataframe['buy_03_trigger'])

        # Protections
        buy_04_protections = [True]
        if self.buy_04_protection__ema_fast.value:
            buy_04_protections.append(dataframe[f"ema_{self.buy_04_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_04_protection__ema_slow.value:
            buy_04_protections.append(dataframe[f"ema_{self.buy_04_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_04_protection__close_above_ema_fast.value:
            buy_04_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_04_protection__close_above_ema_fast_len.value}"])
        if self.buy_04_protection__close_above_ema_slow.value:
            buy_04_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_04_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_04_protection__sma200_rising.value:
            buy_04_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_04_protection__sma200_rising_val.value)))
        if self.buy_04_protection__sma200_1h_rising.value:
            buy_04_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_04_protection__sma200_1h_rising_val.value)))
        if self.buy_04_protection__safe_dips.value:
            buy_04_protections.append(dataframe[f"safe_dips_{self.buy_04_protection__safe_dips_type.value}"])
        if self.buy_04_protection__safe_pump.value:
            buy_04_protections.append(dataframe[f"safe_pump_{self.buy_04_protection__safe_pump_period.value}_{self.buy_04_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_04_logic = []
        buy_04_logic.append(reduce(lambda x, y: x & y, buy_04_protections))
        buy_04_logic.append(dataframe['close'] < dataframe['ema_50'])
        buy_04_logic.append(dataframe['close'] < self.buy_bb20_close_bblowerband_4.value * dataframe['bb20_2_low'])
        buy_04_logic.append(dataframe['volume'] < (dataframe['volume_mean_30'].shift(1) * self.buy_bb20_volume_4.value))
        # Populate
        dataframe.loc[:, 'buy_04_trigger'] = reduce(lambda x, y: x & y, buy_04_logic)
        if self.buy_condition_4_enable.value:
            conditions.append(dataframe['buy_04_trigger'])


        # Protections
        buy_05_protections = [True]
        if self.buy_05_protection__ema_fast.value:
            buy_05_protections.append(dataframe[f"ema_{self.buy_05_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_05_protection__ema_slow.value:
            buy_05_protections.append(dataframe[f"ema_{self.buy_05_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_05_protection__close_above_ema_fast.value:
            buy_05_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_05_protection__close_above_ema_fast_len.value}"])
        if self.buy_05_protection__close_above_ema_slow.value:
            buy_05_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_05_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_05_protection__sma200_rising.value:
            buy_05_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_05_protection__sma200_rising_val.value)))
        if self.buy_05_protection__sma200_1h_rising.value:
            buy_05_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_05_protection__sma200_1h_rising_val.value)))
        if self.buy_05_protection__safe_dips.value:
            buy_05_protections.append(dataframe[f"safe_dips_{self.buy_05_protection__safe_dips_type.value}"])
        if self.buy_05_protection__safe_pump.value:
            buy_05_protections.append(dataframe[f"safe_pump_{self.buy_05_protection__safe_pump_period.value}_{self.buy_05_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)
        buy_05_protections.append(dataframe['close'] > (dataframe['ema_200_1h'] * self.buy_ema_rel_5.value))

        # Logic
        buy_05_logic = []
        buy_05_logic.append(reduce(lambda x, y: x & y, buy_05_protections))
        buy_05_logic.append(dataframe['ema_26'] > dataframe['ema_12'])
        buy_05_logic.append((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_5.value))
        buy_05_logic.append((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))
        buy_05_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_5.value))
        buy_05_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_05_trigger'] = reduce(lambda x, y: x & y, buy_05_logic)
        if self.buy_condition_5_enable.value:
            conditions.append(dataframe['buy_05_trigger'])

        # Protections
        buy_06_protections = [True]
        if self.buy_06_protection__ema_fast.value:
            buy_06_protections.append(dataframe[f"ema_{self.buy_06_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_06_protection__ema_slow.value:
            buy_06_protections.append(dataframe[f"ema_{self.buy_06_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_06_protection__close_above_ema_fast.value:
            buy_06_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_06_protection__close_above_ema_fast_len.value}"])
        if self.buy_06_protection__close_above_ema_slow.value:
            buy_06_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_06_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_06_protection__sma200_rising.value:
            buy_06_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_06_protection__sma200_rising_val.value)))
        if self.buy_06_protection__sma200_1h_rising.value:
            buy_06_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_06_protection__sma200_1h_rising_val.value)))
        if self.buy_06_protection__safe_dips.value:
            buy_06_protections.append(dataframe[f"safe_dips_{self.buy_06_protection__safe_dips_type.value}"])
        if self.buy_06_protection__safe_pump.value:
            buy_06_protections.append(dataframe[f"safe_pump_{self.buy_06_protection__safe_pump_period.value}_{self.buy_06_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_06_logic = []
        buy_06_logic.append(reduce(lambda x, y: x & y, buy_06_protections))
        buy_06_logic.append(dataframe['ema_26'] > dataframe['ema_12'])
        buy_06_logic.append((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_6.value))
        buy_06_logic.append((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))
        buy_06_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_6.value))
        buy_06_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_06_trigger'] = reduce(lambda x, y: x & y, buy_06_logic)
        if self.buy_condition_6_enable.value:
            conditions.append(dataframe['buy_06_trigger'])

        # Protections
        buy_07_protections = [True]
        if self.buy_07_protection__ema_fast.value:
            buy_07_protections.append(dataframe[f"ema_{self.buy_07_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_07_protection__ema_slow.value:
            buy_07_protections.append(dataframe[f"ema_{self.buy_07_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_07_protection__close_above_ema_fast.value:
            buy_07_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_07_protection__close_above_ema_fast_len.value}"])
        if self.buy_07_protection__close_above_ema_slow.value:
            buy_07_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_07_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_07_protection__sma200_rising.value:
            buy_07_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_07_protection__sma200_rising_val.value)))
        if self.buy_07_protection__sma200_1h_rising.value:
            buy_07_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_07_protection__sma200_1h_rising_val.value)))
        if self.buy_07_protection__safe_dips.value:
            buy_07_protections.append(dataframe[f"safe_dips_{self.buy_07_protection__safe_dips_type.value}"])
        if self.buy_07_protection__safe_pump.value:
            buy_07_protections.append(dataframe[f"safe_pump_{self.buy_07_protection__safe_pump_period.value}_{self.buy_07_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_07_logic = []
        buy_07_logic.append(reduce(lambda x, y: x & y, buy_07_protections))
        #buy_07_logic.append(dataframe['volume'].rolling(4).mean() * self.buy_volume_7.value > dataframe['volume'])
        buy_07_logic.append(dataframe['ema_26'] > dataframe['ema_12'])
        buy_07_logic.append((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_7.value))
        buy_07_logic.append((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))
        buy_07_logic.append(dataframe['rsi'] < self.buy_rsi_7.value)
        buy_07_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_07_trigger'] = reduce(lambda x, y: x & y, buy_07_logic)
        if self.buy_condition_7_enable.value:
            conditions.append(dataframe['buy_07_trigger'])

        # Protections
        buy_08_protections = [True]
        if self.buy_08_protection__ema_fast.value:
            buy_08_protections.append(dataframe[f"ema_{self.buy_08_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_08_protection__ema_slow.value:
            buy_08_protections.append(dataframe[f"ema_{self.buy_08_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_08_protection__close_above_ema_fast.value:
            buy_08_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_08_protection__close_above_ema_fast_len.value}"])
        if self.buy_08_protection__close_above_ema_slow.value:
            buy_08_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_08_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_08_protection__sma200_rising.value:
            buy_08_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_08_protection__sma200_rising_val.value)))
        if self.buy_08_protection__sma200_1h_rising.value:
            buy_08_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_08_protection__sma200_1h_rising_val.value)))
        if self.buy_08_protection__safe_dips.value:
            buy_08_protections.append(dataframe[f"safe_dips_{self.buy_08_protection__safe_dips_type.value}"])
        if self.buy_08_protection__safe_pump.value:
            buy_08_protections.append(dataframe[f"safe_pump_{self.buy_08_protection__safe_pump_period.value}_{self.buy_08_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_08_logic = []
        buy_08_logic.append(reduce(lambda x, y: x & y, buy_08_protections))
        buy_08_logic.append(dataframe['rsi'] < self.buy_rsi_8.value)
        buy_08_logic.append(dataframe['volume'] > (dataframe['volume'].shift(1) * self.buy_volume_8.value))
        buy_08_logic.append(dataframe['close'] > dataframe['open'])
        buy_08_logic.append((dataframe['close'] - dataframe['low']) > ((dataframe['close'] - dataframe['open']) * self.buy_tail_diff_8.value))
        buy_08_logic.append(dataframe['volume'] > 0)

        # Populate
        dataframe.loc[:, 'buy_08_trigger'] = reduce(lambda x, y: x & y, buy_08_logic)
        if self.buy_condition_8_enable.value:
            conditions.append(dataframe['buy_08_trigger'])

        # Protections
        buy_09_protections = [True]
        if self.buy_09_protection__ema_fast.value:
            buy_09_protections.append(dataframe[f"ema_{self.buy_09_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_09_protection__ema_slow.value:
            buy_09_protections.append(dataframe[f"ema_{self.buy_09_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_09_protection__close_above_ema_fast.value:
            buy_09_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_09_protection__close_above_ema_fast_len.value}"])
        if self.buy_09_protection__close_above_ema_slow.value:
            buy_09_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_09_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_09_protection__sma200_rising.value:
            buy_09_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_09_protection__sma200_rising_val.value)))
        if self.buy_09_protection__sma200_1h_rising.value:
            buy_09_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_09_protection__sma200_1h_rising_val.value)))
        if self.buy_09_protection__safe_dips.value:
            buy_09_protections.append(dataframe[f"safe_dips_{self.buy_09_protection__safe_dips_type.value}"])
        if self.buy_09_protection__safe_pump.value:
            buy_09_protections.append(dataframe[f"safe_pump_{self.buy_09_protection__safe_pump_period.value}_{self.buy_09_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)
        buy_09_protections.append(dataframe['ema_50'] > dataframe['ema_200'])

        # Logic
        buy_09_logic = []
        buy_09_logic.append(reduce(lambda x, y: x & y, buy_09_protections))
        #buy_09_logic.append(dataframe['volume_mean_4'] * self.buy_volume_9.value > dataframe['volume'])
        buy_09_logic.append(dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_9.value)
        buy_09_logic.append(dataframe['close'] < dataframe['bb20_2_low'] * self.buy_bb_offset_9.value)
        buy_09_logic.append(dataframe['rsi_1h'] > self.buy_rsi_1h_min_9.value)
        buy_09_logic.append(dataframe['rsi_1h'] < self.buy_rsi_1h_max_9.value)
        buy_09_logic.append(dataframe['mfi'] < self.buy_mfi_9.value)
        buy_09_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_09_trigger'] = reduce(lambda x, y: x & y, buy_09_logic)
        if self.buy_condition_9_enable.value:
            conditions.append(dataframe['buy_09_trigger'])

        # Protections
        buy_10_protections = [True]
        if self.buy_10_protection__ema_fast.value:
            buy_10_protections.append(dataframe[f"ema_{self.buy_10_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_10_protection__ema_slow.value:
            buy_10_protections.append(dataframe[f"ema_{self.buy_10_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_10_protection__close_above_ema_fast.value:
            buy_10_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_10_protection__close_above_ema_fast_len.value}"])
        if self.buy_10_protection__close_above_ema_slow.value:
            buy_10_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_10_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_10_protection__sma200_rising.value:
            buy_10_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_10_protection__sma200_rising_val.value)))
        if self.buy_10_protection__sma200_1h_rising.value:
            buy_10_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_10_protection__sma200_1h_rising_val.value)))
        if self.buy_10_protection__safe_dips.value:
            buy_10_protections.append(dataframe[f"safe_dips_{self.buy_10_protection__safe_dips_type.value}"])
        if self.buy_10_protection__safe_pump.value:
            buy_10_protections.append(dataframe[f"safe_pump_{self.buy_10_protection__safe_pump_period.value}_{self.buy_10_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)
        buy_10_protections.append(dataframe['ema_50_1h'] > dataframe['ema_100_1h'])

        # Logic
        buy_10_logic = []
        buy_10_logic.append(reduce(lambda x, y: x & y, buy_10_protections))
        buy_10_logic.append(dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_10.value)
        buy_10_logic.append(dataframe['close'] < dataframe['bb20_2_low'] * self.buy_bb_offset_10.value)
        buy_10_logic.append(dataframe['rsi_1h'] < self.buy_rsi_1h_10.value)
        buy_10_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_10_trigger'] = reduce(lambda x, y: x & y, buy_10_logic)
        if self.buy_condition_10_enable.value:
            conditions.append(dataframe['buy_10_trigger'])

        # Protections
        buy_11_protections = [True]
        if self.buy_11_protection__ema_fast.value:
            buy_11_protections.append(dataframe[f"ema_{self.buy_11_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_11_protection__ema_slow.value:
            buy_11_protections.append(dataframe[f"ema_{self.buy_11_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_11_protection__close_above_ema_fast.value:
            buy_11_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_11_protection__close_above_ema_fast_len.value}"])
        if self.buy_11_protection__close_above_ema_slow.value:
            buy_11_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_11_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_11_protection__sma200_rising.value:
            buy_11_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_11_protection__sma200_rising_val.value)))
        if self.buy_11_protection__sma200_1h_rising.value:
            buy_11_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_11_protection__sma200_1h_rising_val.value)))
        if self.buy_11_protection__safe_dips.value:
            buy_11_protections.append(dataframe[f"safe_dips_{self.buy_11_protection__safe_dips_type.value}"])
        if self.buy_11_protection__safe_pump.value:
            buy_11_protections.append(dataframe[f"safe_pump_{self.buy_11_protection__safe_pump_period.value}_{self.buy_11_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)
        buy_11_protections.append(dataframe['ema_50_1h'] > dataframe['ema_100_1h'])
        buy_11_protections.append(dataframe['safe_pump_36_50_1h'])
        buy_11_protections.append(dataframe['safe_pump_48_100_1h'])

        # Logic
        buy_11_logic = []
        buy_11_logic.append(reduce(lambda x, y: x & y, buy_11_protections))
        buy_11_logic.append(((dataframe['close'] - dataframe['open'].rolling(36).min()) / dataframe['open'].rolling(36).min()) > self.buy_min_inc_11.value)
        buy_11_logic.append(dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_11.value)
        buy_11_logic.append(dataframe['rsi_1h'] > self.buy_rsi_1h_min_11.value)
        buy_11_logic.append(dataframe['rsi_1h'] < self.buy_rsi_1h_max_11.value)
        buy_11_logic.append(dataframe['rsi'] < self.buy_rsi_11.value)
        buy_11_logic.append(dataframe['mfi'] < self.buy_mfi_11.value)
        buy_11_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_11_trigger'] = reduce(lambda x, y: x & y, buy_11_logic)
        if self.buy_condition_11_enable.value:
            conditions.append(dataframe['buy_11_trigger'])

        # Protections
        buy_12_protections = [True]
        if self.buy_12_protection__ema_fast.value:
            buy_12_protections.append(dataframe[f"ema_{self.buy_12_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_12_protection__ema_slow.value:
            buy_12_protections.append(dataframe[f"ema_{self.buy_12_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_12_protection__close_above_ema_fast.value:
            buy_12_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_12_protection__close_above_ema_fast_len.value}"])
        if self.buy_12_protection__close_above_ema_slow.value:
            buy_12_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_12_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_12_protection__sma200_rising.value:
            buy_12_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_12_protection__sma200_rising_val.value)))
        if self.buy_12_protection__sma200_1h_rising.value:
            buy_12_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_12_protection__sma200_1h_rising_val.value)))
        if self.buy_12_protection__safe_dips.value:
            buy_12_protections.append(dataframe[f"safe_dips_{self.buy_12_protection__safe_dips_type.value}"])
        if self.buy_12_protection__safe_pump.value:
            buy_12_protections.append(dataframe[f"safe_pump_{self.buy_12_protection__safe_pump_period.value}_{self.buy_12_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_12_logic = []
        buy_12_logic.append(reduce(lambda x, y: x & y, buy_12_protections))
        #buy_12_logic.append((dataframe['volume_mean_4'] * self.buy_volume_12.value) > dataframe['volume'])
        buy_12_logic.append(dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_12.value)
        buy_12_logic.append(dataframe['ewo'] > self.buy_ewo_12.value)
        buy_12_logic.append(dataframe['rsi'] < self.buy_rsi_12.value)
        buy_12_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_12_trigger'] = reduce(lambda x, y: x & y, buy_12_logic)
        if self.buy_condition_12_enable.value:
            conditions.append(dataframe['buy_12_trigger'])

        # Protections
        buy_13_protections = [True]
        if self.buy_13_protection__ema_fast.value:
            buy_13_protections.append(dataframe[f"ema_{self.buy_13_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_13_protection__ema_slow.value:
            buy_13_protections.append(dataframe[f"ema_{self.buy_13_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_13_protection__close_above_ema_fast.value:
            buy_13_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_13_protection__close_above_ema_fast_len.value}"])
        if self.buy_13_protection__close_above_ema_slow.value:
            buy_13_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_13_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_13_protection__sma200_rising.value:
            buy_13_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_13_protection__sma200_rising_val.value)))
        if self.buy_13_protection__sma200_1h_rising.value:
            buy_13_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_13_protection__sma200_1h_rising_val.value)))
        if self.buy_13_protection__safe_dips.value:
            buy_13_protections.append(dataframe[f"safe_dips_{self.buy_13_protection__safe_dips_type.value}"])
        if self.buy_13_protection__safe_pump.value:
            buy_13_protections.append(dataframe[f"safe_pump_{self.buy_13_protection__safe_pump_period.value}_{self.buy_13_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)
        buy_13_protections.append(dataframe['ema_50_1h'] > dataframe['ema_100_1h'])
        #buy_13_protections.append(dataframe['safe_pump_36_loose_1h'])

        # Logic
        buy_13_logic = []
        buy_13_logic.append(reduce(lambda x, y: x & y, buy_13_protections))
        #buy_13_logic.append((dataframe['volume_mean_4'] * self.buy_volume_13.value) > dataframe['volume'])
        buy_13_logic.append(dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_13.value)
        buy_13_logic.append(dataframe['ewo'] < self.buy_ewo_13.value)
        buy_13_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_13_trigger'] = reduce(lambda x, y: x & y, buy_13_logic)
        if self.buy_condition_13_enable.value:
            conditions.append(dataframe['buy_13_trigger'])

        # Protections
        buy_14_protections = [True]
        if self.buy_14_protection__ema_fast.value:
            buy_14_protections.append(dataframe[f"ema_{self.buy_14_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_14_protection__ema_slow.value:
            buy_14_protections.append(dataframe[f"ema_{self.buy_14_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_14_protection__close_above_ema_fast.value:
            buy_14_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_14_protection__close_above_ema_fast_len.value}"])
        if self.buy_14_protection__close_above_ema_slow.value:
            buy_14_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_14_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_14_protection__sma200_rising.value:
            buy_14_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_14_protection__sma200_rising_val.value)))
        if self.buy_14_protection__sma200_1h_rising.value:
            buy_14_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_14_protection__sma200_1h_rising_val.value)))
        if self.buy_14_protection__safe_dips.value:
            buy_14_protections.append(dataframe[f"safe_dips_{self.buy_14_protection__safe_dips_type.value}"])
        if self.buy_14_protection__safe_pump.value:
            buy_14_protections.append(dataframe[f"safe_pump_{self.buy_14_protection__safe_pump_period.value}_{self.buy_14_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_14_logic = []
        buy_14_logic.append(reduce(lambda x, y: x & y, buy_14_protections))
        #buy_14_logic.append(dataframe['volume_mean_4'] * self.buy_volume_14.value > dataframe['volume'])
        buy_14_logic.append(dataframe['ema_26'] > dataframe['ema_12'])
        buy_14_logic.append((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_14.value))
        buy_14_logic.append((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))
        buy_14_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_14.value))
        buy_14_logic.append(dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_14.value)
        buy_14_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_14_trigger'] = reduce(lambda x, y: x & y, buy_14_logic)
        if self.buy_condition_14_enable.value:
            conditions.append(dataframe['buy_14_trigger'])

        # Protections
        buy_15_protections = [True]
        if self.buy_15_protection__ema_fast.value:
            buy_15_protections.append(dataframe[f"ema_{self.buy_15_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_15_protection__ema_slow.value:
            buy_15_protections.append(dataframe[f"ema_{self.buy_15_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_15_protection__close_above_ema_fast.value:
            buy_15_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_15_protection__close_above_ema_fast_len.value}"])
        if self.buy_15_protection__close_above_ema_slow.value:
            buy_15_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_15_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_15_protection__sma200_rising.value:
            buy_15_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_15_protection__sma200_rising_val.value)))
        if self.buy_15_protection__sma200_1h_rising.value:
            buy_15_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_15_protection__sma200_1h_rising_val.value)))
        if self.buy_15_protection__safe_dips.value:
            buy_15_protections.append(dataframe[f"safe_dips_{self.buy_15_protection__safe_dips_type.value}"])
        if self.buy_15_protection__safe_pump.value:
            buy_15_protections.append(dataframe[f"safe_pump_{self.buy_15_protection__safe_pump_period.value}_{self.buy_15_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)
        buy_15_protections.append(dataframe['close'] > dataframe['ema_200_1h'] * self.buy_ema_rel_15.value)

        # Logic
        buy_15_logic = []
        buy_15_logic.append(reduce(lambda x, y: x & y, buy_15_protections))
        buy_15_logic.append(dataframe['ema_26'] > dataframe['ema_12'])
        buy_15_logic.append((dataframe['ema_26'] - dataframe['ema_12']) > (dataframe['open'] * self.buy_ema_open_mult_15.value))
        buy_15_logic.append((dataframe['ema_26'].shift() - dataframe['ema_12'].shift()) > (dataframe['open'] / 100))
        buy_15_logic.append(dataframe['rsi'] < self.buy_rsi_15.value)
        buy_15_logic.append(dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_15.value)
        buy_15_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_15_trigger'] = reduce(lambda x, y: x & y, buy_15_logic)
        if self.buy_condition_15_enable.value:
            conditions.append(dataframe['buy_15_trigger'])

        # Protections
        buy_16_protections = [True]
        if self.buy_16_protection__ema_fast.value:
            buy_16_protections.append(dataframe[f"ema_{self.buy_16_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_16_protection__ema_slow.value:
            buy_16_protections.append(dataframe[f"ema_{self.buy_16_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_16_protection__close_above_ema_fast.value:
            buy_16_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_16_protection__close_above_ema_fast_len.value}"])
        if self.buy_16_protection__close_above_ema_slow.value:
            buy_16_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_16_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_16_protection__sma200_rising.value:
            buy_16_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_16_protection__sma200_rising_val.value)))
        if self.buy_16_protection__sma200_1h_rising.value:
            buy_16_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_16_protection__sma200_1h_rising_val.value)))
        if self.buy_16_protection__safe_dips.value:
            buy_16_protections.append(dataframe[f"safe_dips_{self.buy_16_protection__safe_dips_type.value}"])
        if self.buy_16_protection__safe_pump.value:
            buy_16_protections.append(dataframe[f"safe_pump_{self.buy_16_protection__safe_pump_period.value}_{self.buy_16_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_16_logic = []
        buy_16_logic.append(reduce(lambda x, y: x & y, buy_16_protections))
        #buy_16_logic.append((dataframe['volume_mean_4'] * self.buy_volume_16.value) > dataframe['volume'])
        buy_16_logic.append(dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_16.value)
        buy_16_logic.append(dataframe['ewo'] > self.buy_ewo_16.value)
        buy_16_logic.append(dataframe['rsi'] < self.buy_rsi_16.value)
        buy_16_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_16_trigger'] = reduce(lambda x, y: x & y, buy_16_logic)
        if self.buy_condition_16_enable.value:
            conditions.append(dataframe['buy_16_trigger'])

        # Protections
        buy_17_protections = [True]
        if self.buy_17_protection__ema_fast.value:
            buy_17_protections.append(dataframe[f"ema_{self.buy_17_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_17_protection__ema_slow.value:
            buy_17_protections.append(dataframe[f"ema_{self.buy_17_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_17_protection__close_above_ema_fast.value:
            buy_17_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_17_protection__close_above_ema_fast_len.value}"])
        if self.buy_17_protection__close_above_ema_slow.value:
            buy_17_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_17_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_17_protection__sma200_rising.value:
            buy_17_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_17_protection__sma200_rising_val.value)))
        if self.buy_17_protection__sma200_1h_rising.value:
            buy_17_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_17_protection__sma200_1h_rising_val.value)))
        if self.buy_17_protection__safe_dips.value:
            buy_17_protections.append(dataframe[f"safe_dips_{self.buy_17_protection__safe_dips_type.value}"])
        if self.buy_17_protection__safe_pump.value:
            buy_17_protections.append(dataframe[f"safe_pump_{self.buy_17_protection__safe_pump_period.value}_{self.buy_17_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_17_logic = []
        buy_17_logic.append(reduce(lambda x, y: x & y, buy_17_protections))
        buy_17_logic.append(dataframe['close'] < dataframe['ema_20'] * self.buy_ma_offset_17.value)
        buy_17_logic.append(dataframe['ewo'] < self.buy_ewo_17.value)
        buy_17_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_17_trigger'] = reduce(lambda x, y: x & y, buy_17_logic)
        if self.buy_condition_17_enable.value:
            conditions.append(dataframe['buy_17_trigger'])

        # Protections
        buy_18_protections = [True]
        if self.buy_18_protection__ema_fast.value:
            buy_18_protections.append(dataframe[f"ema_{self.buy_18_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_18_protection__ema_slow.value:
            buy_18_protections.append(dataframe[f"ema_{self.buy_18_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_18_protection__close_above_ema_fast.value:
            buy_18_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_18_protection__close_above_ema_fast_len.value}"])
        if self.buy_18_protection__close_above_ema_slow.value:
            buy_18_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_18_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_18_protection__sma200_rising.value:
            buy_18_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_18_protection__sma200_rising_val.value)))
        if self.buy_18_protection__sma200_1h_rising.value:
            buy_18_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_18_protection__sma200_1h_rising_val.value)))
        if self.buy_18_protection__safe_dips.value:
            buy_18_protections.append(dataframe[f"safe_dips_{self.buy_18_protection__safe_dips_type.value}"])
        if self.buy_18_protection__safe_pump.value:
            buy_18_protections.append(dataframe[f"safe_pump_{self.buy_18_protection__safe_pump_period.value}_{self.buy_18_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)
        #buy_18_protections.append(dataframe['ema_100'] > dataframe['ema_200'])
        buy_18_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(20))
        buy_18_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(36))

        # Logic
        buy_18_logic = []
        buy_18_logic.append(reduce(lambda x, y: x & y, buy_18_protections))
        #buy_18_logic.append((dataframe['volume_mean_4'] * self.buy_volume_18.value) > dataframe['volume'])
        buy_18_logic.append(dataframe['rsi'] < self.buy_rsi_18.value)
        buy_18_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_18.value))
        buy_18_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_18_trigger'] = reduce(lambda x, y: x & y, buy_18_logic)
        if self.buy_condition_18_enable.value:
            conditions.append(dataframe['buy_18_trigger'])

        # Protections
        buy_19_protections = [True]
        if self.buy_19_protection__ema_fast.value:
            buy_19_protections.append(dataframe[f"ema_{self.buy_19_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_19_protection__ema_slow.value:
            buy_19_protections.append(dataframe[f"ema_{self.buy_19_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_19_protection__close_above_ema_fast.value:
            buy_19_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_19_protection__close_above_ema_fast_len.value}"])
        if self.buy_19_protection__close_above_ema_slow.value:
            buy_19_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_19_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_19_protection__sma200_rising.value:
            buy_19_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_19_protection__sma200_rising_val.value)))
        if self.buy_19_protection__sma200_1h_rising.value:
            buy_19_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_19_protection__sma200_1h_rising_val.value)))
        if self.buy_19_protection__safe_dips.value:
            buy_19_protections.append(dataframe[f"safe_dips_{self.buy_19_protection__safe_dips_type.value}"])
        if self.buy_19_protection__safe_pump.value:
            buy_19_protections.append(dataframe[f"safe_pump_{self.buy_19_protection__safe_pump_period.value}_{self.buy_19_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)
        buy_19_protections.append(dataframe['ema_50_1h'] > dataframe['ema_200_1h'])

        # Logic
        buy_19_logic = []
        buy_19_logic.append(reduce(lambda x, y: x & y, buy_19_protections))
        buy_19_logic.append(dataframe['close'].shift(1) > dataframe['ema_100_1h'])
        buy_19_logic.append(dataframe['low'] < dataframe['ema_100_1h'])
        buy_19_logic.append(dataframe['close'] > dataframe['ema_100_1h'])
        buy_19_logic.append(dataframe['rsi_1h'] > self.buy_rsi_1h_min_19.value)
        buy_19_logic.append(dataframe['chop'] < self.buy_chop_min_19.value)
        buy_19_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_19_trigger'] = reduce(lambda x, y: x & y, buy_19_logic)
        if self.buy_condition_19_enable.value:
            conditions.append(dataframe['buy_19_trigger'])

        # Protections
        buy_20_protections = [True]
        if self.buy_20_protection__ema_fast.value:
            buy_20_protections.append(dataframe[f"ema_{self.buy_20_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_20_protection__ema_slow.value:
            buy_20_protections.append(dataframe[f"ema_{self.buy_20_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_20_protection__close_above_ema_fast.value:
            buy_20_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_20_protection__close_above_ema_fast_len.value}"])
        if self.buy_20_protection__close_above_ema_slow.value:
            buy_20_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_20_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_20_protection__sma200_rising.value:
            buy_20_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_20_protection__sma200_rising_val.value)))
        if self.buy_20_protection__sma200_1h_rising.value:
            buy_20_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_20_protection__sma200_1h_rising_val.value)))
        if self.buy_20_protection__safe_dips.value:
            buy_20_protections.append(dataframe[f"safe_dips_{self.buy_20_protection__safe_dips_type.value}"])
        if self.buy_20_protection__safe_pump.value:
            buy_20_protections.append(dataframe[f"safe_pump_{self.buy_20_protection__safe_pump_period.value}_{self.buy_20_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_20_logic = []
        buy_20_logic.append(reduce(lambda x, y: x & y, buy_20_protections))
        #buy_20_logic.append((dataframe['volume_mean_4'] * self.buy_volume_20.value) > dataframe['volume'])
        buy_20_logic.append(dataframe['rsi'] < self.buy_rsi_20.value)
        buy_20_logic.append(dataframe['rsi_1h'] < self.buy_rsi_1h_20.value)
        buy_20_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_20_trigger'] = reduce(lambda x, y: x & y, buy_20_logic)
        if self.buy_condition_20_enable.value:
            conditions.append(dataframe['buy_20_trigger'])

        # Protections
        buy_21_protections = [True]
        if self.buy_21_protection__ema_fast.value:
            buy_21_protections.append(dataframe[f"ema_{self.buy_21_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_21_protection__ema_slow.value:
            buy_21_protections.append(dataframe[f"ema_{self.buy_21_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_21_protection__close_above_ema_fast.value:
            buy_21_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_21_protection__close_above_ema_fast_len.value}"])
        if self.buy_21_protection__close_above_ema_slow.value:
            buy_21_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_21_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_21_protection__sma200_rising.value:
            buy_21_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_21_protection__sma200_rising_val.value)))
        if self.buy_21_protection__sma200_1h_rising.value:
            buy_21_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_21_protection__sma200_1h_rising_val.value)))
        if self.buy_21_protection__safe_dips.value:
            buy_21_protections.append(dataframe[f"safe_dips_{self.buy_21_protection__safe_dips_type.value}"])
        if self.buy_21_protection__safe_pump.value:
            buy_21_protections.append(dataframe[f"safe_pump_{self.buy_21_protection__safe_pump_period.value}_{self.buy_21_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_21_logic = []
        buy_21_logic.append(reduce(lambda x, y: x & y, buy_21_protections))
        #buy_21_logic.append((dataframe['volume_mean_4'] * self.buy_volume_21.value) > dataframe['volume'])
        buy_21_logic.append(dataframe['rsi'] < self.buy_rsi_21.value)
        buy_21_logic.append(dataframe['rsi_1h'] < self.buy_rsi_1h_21.value)
        buy_21_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_21_trigger'] = reduce(lambda x, y: x & y, buy_21_logic)
        if self.buy_condition_21_enable.value:
            conditions.append(dataframe['buy_21_trigger'])

        # Protections
        buy_22_protections = [True]
        if self.buy_22_protection__ema_fast.value:
            buy_22_protections.append(dataframe[f"ema_{self.buy_22_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_22_protection__ema_slow.value:
            buy_22_protections.append(dataframe[f"ema_{self.buy_22_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_22_protection__close_above_ema_fast.value:
            buy_22_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_22_protection__close_above_ema_fast_len.value}"])
        if self.buy_22_protection__close_above_ema_slow.value:
            buy_22_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_22_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_22_protection__sma200_rising.value:
            buy_22_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_22_protection__sma200_rising_val.value)))
        if self.buy_22_protection__sma200_1h_rising.value:
            buy_22_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_22_protection__sma200_1h_rising_val.value)))
        if self.buy_22_protection__safe_dips.value:
            buy_22_protections.append(dataframe[f"safe_dips_{self.buy_22_protection__safe_dips_type.value}"])
        if self.buy_22_protection__safe_pump.value:
            buy_22_protections.append(dataframe[f"safe_pump_{self.buy_22_protection__safe_pump_period.value}_{self.buy_22_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)
        buy_22_protections.append(dataframe['ema_100_1h'] > dataframe['ema_100_1h'].shift(12))
        buy_22_protections.append(dataframe['ema_200_1h'] > dataframe['ema_200_1h'].shift(36))

        # Logic
        buy_22_logic = []
        buy_22_logic.append(reduce(lambda x, y: x & y, buy_22_protections))
        buy_22_logic.append((dataframe['volume_mean_4'] * self.buy_volume_22.value) > dataframe['volume'])
        buy_22_logic.append(dataframe['close'] < dataframe['sma_30'] * self.buy_ma_offset_22.value)
        buy_22_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_22.value))
        buy_22_logic.append(dataframe['ewo'] > self.buy_ewo_22.value)
        buy_22_logic.append(dataframe['rsi'] < self.buy_rsi_22.value)
        buy_22_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_22_trigger'] = reduce(lambda x, y: x & y, buy_22_logic)
        if self.buy_condition_22_enable.value:
            conditions.append(dataframe['buy_22_trigger'])

            # Protections
        buy_23_protections = [True]
        if self.buy_23_protection__ema_fast.value:
            buy_23_protections.append(dataframe[f"ema_{self.buy_23_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_23_protection__ema_slow.value:
            buy_23_protections.append(dataframe[f"ema_{self.buy_23_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_23_protection__close_above_ema_fast.value:
            buy_23_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_23_protection__close_above_ema_fast_len.value}"])
        if self.buy_23_protection__close_above_ema_slow.value:
            buy_23_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_23_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_23_protection__sma200_rising.value:
            buy_23_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_23_protection__sma200_rising_val.value)))
        if self.buy_23_protection__sma200_1h_rising.value:
            buy_23_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_23_protection__sma200_1h_rising_val.value)))
        if self.buy_23_protection__safe_dips.value:
            buy_23_protections.append(dataframe[f"safe_dips_{self.buy_23_protection__safe_dips_type.value}"])
        if self.buy_23_protection__safe_pump.value:
            buy_23_protections.append(dataframe[f"safe_pump_{self.buy_23_protection__safe_pump_period.value}_{self.buy_23_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_23_logic = []
        buy_23_logic.append(reduce(lambda x, y: x & y, buy_23_protections))
        buy_23_logic.append(dataframe['close'] < (dataframe['bb20_2_low'] * self.buy_bb_offset_23.value))
        buy_23_logic.append(dataframe['ewo'] > self.buy_ewo_23.value)
        buy_23_logic.append(dataframe['rsi'] < self.buy_rsi_23.value)
        buy_23_logic.append(dataframe['rsi_1h'] < self.buy_rsi_1h_23.value)
        buy_23_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_23_trigger'] = reduce(lambda x, y: x & y, buy_23_logic)
        if self.buy_condition_23_enable.value:
            conditions.append(dataframe['buy_23_trigger'])


        # Protections
        buy_24_protections = [True]
        if self.buy_24_protection__ema_fast.value:
            buy_24_protections.append(dataframe[f"ema_{self.buy_24_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_24_protection__ema_slow.value:
            buy_24_protections.append(dataframe[f"ema_{self.buy_24_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_24_protection__close_above_ema_fast.value:
            buy_24_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_24_protection__close_above_ema_fast_len.value}"])
        if self.buy_24_protection__close_above_ema_slow.value:
            buy_24_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_24_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_24_protection__sma200_rising.value:
            buy_24_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_24_protection__sma200_rising_val.value)))
        if self.buy_24_protection__sma200_1h_rising.value:
            buy_24_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_24_protection__sma200_1h_rising_val.value)))
        if self.buy_24_protection__safe_dips.value:
            buy_24_protections.append(dataframe[f"safe_dips_{self.buy_24_protection__safe_dips_type.value}"])
        if self.buy_24_protection__safe_pump.value:
            buy_24_protections.append(dataframe[f"safe_pump_{self.buy_24_protection__safe_pump_period.value}_{self.buy_24_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_24_logic = []
        buy_24_logic.append(reduce(lambda x, y: x & y, buy_24_protections))
        buy_24_logic.append(dataframe['ema_12_1h'].shift(12) < dataframe['ema_35_1h'].shift(12))
        buy_24_logic.append(dataframe['ema_12_1h'].shift(12) < dataframe['ema_35_1h'].shift(12))
        buy_24_logic.append(dataframe['ema_12_1h'] > dataframe['ema_35_1h'])
        buy_24_logic.append(dataframe['cmf_1h'].shift(12) < 0)
        buy_24_logic.append(dataframe['cmf_1h'] > 0)
        buy_24_logic.append(dataframe['rsi'] < self.buy_24_rsi_max.value)
        buy_24_logic.append(dataframe['rsi_1h'] > self.buy_24_rsi_1h_min.value)
        buy_24_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:, 'buy_24_trigger'] = reduce(lambda x, y: x & y, buy_24_logic)
        if self.buy_condition_24_enable.value:
            conditions.append(dataframe['buy_24_trigger'])

        # Protections
        buy_26_protections = [True]
        if self.buy_26_protection__ema_fast.value:
            buy_26_protections.append(dataframe[f"ema_{self.buy_26_protection__ema_fast_len.value}"] > dataframe['ema_200'])
        if self.buy_26_protection__ema_slow.value:
            buy_26_protections.append(dataframe[f"ema_{self.buy_26_protection__ema_slow_len.value}_1h"] > dataframe['ema_200_1h'])
        if self.buy_26_protection__close_above_ema_fast.value:
            buy_26_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_26_protection__close_above_ema_fast_len.value}"])
        if self.buy_26_protection__close_above_ema_slow.value:
            buy_26_protections.append(dataframe['close'] > dataframe[f"ema_{self.buy_26_protection__close_above_ema_slow_len.value}_1h"])
        if self.buy_26_protection__sma200_rising.value:
            buy_26_protections.append(dataframe['sma_200'] > dataframe['sma_200'].shift(int(self.buy_26_protection__sma200_rising_val.value)))
        if self.buy_26_protection__sma200_1h_rising.value:
            buy_26_protections.append(dataframe['sma_200_1h'] > dataframe['sma_200_1h'].shift(int(self.buy_26_protection__sma200_1h_rising_val.value)))
        if self.buy_26_protection__safe_dips.value:
            buy_26_protections.append(dataframe[f"safe_dips_{self.buy_26_protection__safe_dips_type.value}"])
        if self.buy_26_protection__safe_pump.value:
            buy_26_protections.append(dataframe[f"safe_pump_{self.buy_26_protection__safe_pump_period.value}_{self.buy_26_protection__safe_pump_type.value}_1h"])
        # Non-Standard protections (add below)

        # Logic
        buy_26_logic = []
        buy_26_logic.append(reduce(lambda x, y: x & y, buy_26_protections))
        buy_26_logic.append(dataframe['close'] < (dataframe['zema'] * self.buy_26_zema_low_offset.value))
        buy_26_logic.append(dataframe['volume'] > 0)
        # Populate
        dataframe.loc[:,'buy_26_trigger'] = reduce(lambda x, y: x & y, buy_26_logic)
        if self.buy_condition_26_enable.value:
            conditions.append(dataframe['buy_26_trigger'])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'buy'
            ] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        conditions.append(
            (
                self.sell_condition_1_enable.value &

                (dataframe['rsi'] > self.sell_rsi_bb_1.value) &
                (dataframe['close'] > dataframe['bb20_2_upp']) &
                (dataframe['close'].shift(1) > dataframe['bb20_2_upp'].shift(1)) &
                (dataframe['close'].shift(2) > dataframe['bb20_2_upp'].shift(2)) &
                (dataframe['close'].shift(3) > dataframe['bb20_2_upp'].shift(3)) &
                (dataframe['close'].shift(4) > dataframe['bb20_2_upp'].shift(4)) &
                (dataframe['close'].shift(5) > dataframe['bb20_2_upp'].shift(5)) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_2_enable.value &

                (dataframe['rsi'] > self.sell_rsi_bb_2.value) &
                (dataframe['close'] > dataframe['bb20_2_upp']) &
                (dataframe['close'].shift(1) > dataframe['bb20_2_upp'].shift(1)) &
                (dataframe['close'].shift(2) > dataframe['bb20_2_upp'].shift(2)) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_3_enable.value &

                (dataframe['rsi'] > self.sell_rsi_main_3.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_4_enable.value &

                (dataframe['rsi'] > self.sell_dual_rsi_rsi_4.value) &
                (dataframe['rsi_1h'] > self.sell_dual_rsi_rsi_1h_4.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_6_enable.value &

                (dataframe['close'] < dataframe['ema_200']) &
                (dataframe['close'] > dataframe['ema_50']) &
                (dataframe['rsi'] > self.sell_rsi_under_6.value) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_7_enable.value &

                (dataframe['rsi_1h'] > self.sell_rsi_1h_7.value) &
                qtpylib.crossed_below(dataframe['ema_12'], dataframe['ema_26']) &
                (dataframe['volume'] > 0)
            )
        )

        conditions.append(
            (
                self.sell_condition_8_enable.value &

                (dataframe['close'] > dataframe['bb20_2_upp_1h'] * self.sell_bb_relative_8.value) &

                (dataframe['volume'] > 0)
            )
        )

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'sell'
            ] = 1

        return dataframe


# Elliot Wave Oscillator
def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.EMA(df, timeperiod=sma1_length)
    sma2 = ta.EMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif

# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False):
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= df['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / df['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')
    
def PMAX(df, period, multiplier, length, MAtype, src):
        
    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)
    

    mavalue = 'MA_' + str(MAtype) + '_' + str(length)
    atr = 'ATR_' + str(period)
    pm = 'pm_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)
    pmx = 'pmX_' + str(period) + '_' + str(multiplier) + '_' + str(length) + '_' + str(MAtype)

    

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema          
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4
    if MAtype == 1:
        df[mavalue] = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        df[mavalue] = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        df[mavalue] = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        df[mavalue] = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        df[mavalue] = VIDYA(df, length=length)
    elif MAtype == 6:
        df[mavalue] = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        df[mavalue] = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        df[mavalue] = vwma(df, length)
    elif MAtype == 9:
        df[mavalue] = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = df[mavalue] + ((multiplier/10) * df[atr])
    df['basic_lb'] = df[mavalue] - ((multiplier/10) * df[atr])
    

    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)
    mavalues = df[mavalue].values

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalues[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalues[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb


    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalues[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalues[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalues[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalues[i] < final_lb[i]) else 0.00)

    df[pm] = pm_arr
    
    # Mark the trend direction up/down
    df[pmx] = np.where((df[pm] > 0.00), np.where((df[mavalue] < df[pm]), 'down',  'up'), np.NaN)
    # Remove basic and final bands from the columns
    df.drop(['basic_ub', 'basic_lb', 'final_ub', 'final_lb'], inplace=True, axis=1)

    df.fillna(0, inplace=True)

    return df


class IndicatorMixin:
    """Util mixin indicator class"""

    _fillna = False

    def _check_fillna(self, series: pd.Series, value: int = 0) -> pd.Series:
        """Check if fillna flag is True.
        Args:
            series(pandas.Series): dataset 'Close' column.
            value(int): value to fill gaps; if -1 fill values using 'backfill' mode.
        Returns:
            pandas.Series: New feature generated.
        """
        if self._fillna:
            series_output = series.copy(deep=False)
            series_output = series_output.replace([np.inf, -np.inf], np.nan)
            if isinstance(value, int) and value == -1:
                series = series_output.fillna(method="ffill").fillna(value=-1)
            else:
                series = series_output.fillna(method="ffill").fillna(value)
        return series

    @staticmethod
    def _true_range(
        high: pd.Series, low: pd.Series, prev_close: pd.Series
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.DataFrame(data={"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        return true_range

class TSIIndicator(IndicatorMixin):
    """True strength index (TSI)
    Shows both trend direction and overbought/oversold conditions.
    https://school.stockcharts.com/doku.php?id=technical_indicators:true_strength_index
    Args:
        close(pandas.Series): dataset 'Close' column.
        window_slow(int): high period.
        window_fast(int): low period.
        fillna(bool): if True, fill nan values.
    """

    def __init__(
        self,
        close: pd.Series,
        window_slow: int = 25,
        window_fast: int = 13,
        fillna: bool = False,
    ):
        self._close = close
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._fillna = fillna
        self._run()

    def _run(self):
        diff_close = self._close - self._close.shift(1)
        min_periods_r = 0 if self._fillna else self._window_slow
        min_periods_s = 0 if self._fillna else self._window_fast
        smoothed = (
            diff_close.ewm(
                span=self._window_slow, min_periods=min_periods_r, adjust=False
            )
            .mean()
            .ewm(span=self._window_fast, min_periods=min_periods_s, adjust=False)
            .mean()
        )
        smoothed_abs = (
            abs(diff_close)
            .ewm(span=self._window_slow, min_periods=min_periods_r, adjust=False)
            .mean()
            .ewm(span=self._window_fast, min_periods=min_periods_s, adjust=False)
            .mean()
        )
        self._tsi = smoothed / smoothed_abs
        self._tsi *= 100

    def tsi(self) -> pd.Series:
        """True strength index (TSI)
        Returns:
            pandas.Series: New feature generated.
        """
        tsi_series = self._check_fillna(self._tsi, value=0)
        return pd.Series(tsi_series, name="tsi")
