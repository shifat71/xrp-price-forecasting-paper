
from market_management import get_current_market_price 
import logging  

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
    one_day_3_min_candles = get_current_market_price(
        session=SESSION,
        symbol=SYMBOL,
        interval="3", 
        limit="480",
        start_time="1749978000000",
        end_time="1750068000000"
        )
    print(one_day_3_min_candles)
if __name__ == "__main__":
    main()