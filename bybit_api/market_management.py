import logging

import pandas as pd
import time


def get_current_market_price(session, symbol, interval="1", limit="1", start_time=None, end_time=None ):
    try:
        response = session.get_mark_price_kline(
            category="linear",
            symbol=symbol,
            interval=str(interval),   
            limit=limit,
            start=start_time,
            end=end_time
        )   
        data = response['result']['list']
        if data:
            df = pd.DataFrame(data)
            df = df.iloc[:, 0:5]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close']
            df = df.astype(float)
            return df
        else:
            logging.error(f"Invalid kline response: {response}")
            return None
    except Exception as e:
        logging.error(f"Error fetching klines: {e}")
        return None
    time.sleep(1)
     

def get_min_klines(session, symbol, interval, limit):
    try:
        response = session.get_mark_price_kline(
            category="linear",
            symbol=symbol,
            interval=str(interval),   
            limit=limit        
        )
        data = response['result']['list']
        if data:
            df = pd.DataFrame(data)
            df = df.iloc[:, 0:5]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close']
            df = df.astype(float)
            return df
        else:
            logging.error(f"Invalid kline response: {response}")
            return None
    except Exception as e:
        logging.error(f"Error fetching klines: {e}")
        return None

