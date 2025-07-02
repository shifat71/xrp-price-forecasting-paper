from market_management import get_current_market_price, save_market_data_to_json
import logging  
from datetime import datetime

from clients import client1
 

# Constants
SYMBOL = "XRPUSDT"  

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading.log'),
        logging.StreamHandler()
    ]
)

SESSION = client1


def main(): 
    st = "1751151600000"
    et = "1751155200000"

    one_day_3_min_candles = get_current_market_price(
        session=SESSION,
        symbol=SYMBOL,
        interval="3", 
        limit="20",
        start_time=st,
        end_time=et
        )
    
    if one_day_3_min_candles is not None:
        print("Market data retrieved successfully:")
        print(one_day_3_min_candles.head())
        
        # Save data to JSON file with additional metadata
        additional_info = {
            "interval": "3min",
<<<<<<< HEAD
            "limit": "20",
            "start_time": st,
            "end_time": et
=======
            "limit": "480",
            "start_time": "1751151600000",
            "end_time": "1751155200000"
>>>>>>> 48c8f6f300e025fb8e46d6b7456d66974dcd9110
        }
        
        json_filename = save_market_data_to_json(
            data=one_day_3_min_candles,
            symbol=SYMBOL,
            additional_info=additional_info
        )
        
        if json_filename:
            print(f"Data saved to: {json_filename}")
    else:
        print("Failed to retrieve market data")

if __name__ == "__main__":
    main()