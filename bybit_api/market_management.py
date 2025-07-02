import logging
import json
import os
from datetime import datetime

import pandas as pd
import time


def save_market_data_to_json(data, symbol, filename=None, additional_info=None):
    """
    Save market data to JSON file with proper formatting
    
    Args:
        data (pd.DataFrame): Market data DataFrame
        symbol (str): Trading symbol
        filename (str, optional): Custom filename
        additional_info (dict, optional): Additional information to include
    
    Returns:
        str: Filename if successful, None if failed
    """
    if data is None or data.empty:
        logging.error("No data to save")
        return None
    
    # Create filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_market_data_{timestamp}.json"
    
    # Ensure the dataset directory exists
    dataset_dir = "../dataset/testing_data"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    # Full path for the JSON file
    json_path = os.path.join(dataset_dir, filename)
    
    # Create the data structure
    data_dict = {
        "metadata": {
            "symbol": symbol,
            "timestamp_generated": datetime.now().isoformat(),
            "data_count": len(data),
            "columns": list(data.columns),
            "date_range": {
                "start_timestamp": int(data['timestamp'].min()),
                "end_timestamp": int(data['timestamp'].max()),
                "start_date": datetime.fromtimestamp(data['timestamp'].min() / 1000).isoformat(),
                "end_date": datetime.fromtimestamp(data['timestamp'].max() / 1000).isoformat()
            }
        },
        "market_data": []
    }
    
    # Add additional info if provided
    if additional_info:
        data_dict["metadata"].update(additional_info)
        
        # Convert timestamp strings in additional_info to natural dates if they exist
        if "start_time" in additional_info:
            try:
                start_ms = int(additional_info["start_time"])
                data_dict["metadata"]["start_time_readable"] = datetime.fromtimestamp(start_ms / 1000).isoformat()
            except (ValueError, TypeError):
                pass
        
        if "end_time" in additional_info:
            try:
                end_ms = int(additional_info["end_time"])
                data_dict["metadata"]["end_time_readable"] = datetime.fromtimestamp(end_ms / 1000).isoformat()
            except (ValueError, TypeError):
                pass
    
    # Convert each row to a properly formatted dictionary
    for index, row in data.iterrows():
        # Convert millisecond timestamp to readable date formats
        timestamp_ms = int(row['timestamp'])
        dt = datetime.fromtimestamp(timestamp_ms / 1000)
        
        record = {
            "timestamp_ms": timestamp_ms,
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "date": dt.strftime("%Y-%m-%d"),
            "time": dt.strftime("%H:%M:%S"),
            "iso_format": dt.isoformat(),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close'])
        }
        data_dict["market_data"].append(record)
    
    # Save to JSON file with proper formatting
    try:
        with open(json_path, 'w') as json_file:
            json.dump(data_dict, json_file, indent=2, ensure_ascii=False)
        logging.info(f"Data successfully saved to {json_path}")
        return json_path
    except Exception as e:
        logging.error(f"Error saving data to JSON: {e}")
        return None


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
     
 