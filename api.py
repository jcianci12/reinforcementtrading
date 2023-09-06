import csv
import datetime
import decimal
import hashlib
import hmac
from logging import Logger
import os
import shutil
import time
import uuid
import pandas as pd
import requests
from KEYS import API_KEY, API_SECRET
from functions.interval_map import *
from functions.logger import logger

from KEYS import API_KEY, API_SECRET
import ccxt


exchange = ccxt.binance({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'enableRateLimit': True,
    "timeout":10000000
})
# very important set spot as default type
exchange.options['defaultType'] = 'spot'
exchange.load_markets()
DELAY = 5
TEST_URL = 'https://www.google.com'





def get_min_qty_binance(symbol: str) -> float:
    markets = exchange.load_markets()
    market = markets[symbol]
    min_qty = market['limits']['amount']['min']
    return min_qty


def convert_interval_to_timespan(interval):
    minutes = interval_map[interval]
    return datetime.timedelta(minutes=minutes)


def get_intervals(start_date, end_date, interval):
    interval_timespan = convert_interval_to_timespan(interval)
    time_span_seconds = interval_timespan.total_seconds()
    total_seconds = (end_date - start_date).total_seconds()

    total_intervals = total_seconds // time_span_seconds
    print("there are ", datetime.timedelta(
        seconds=total_seconds).days, "total days in the range.")

    print(f'total_intervals: {total_intervals}')
    if total_intervals > 200:
        intervals = []
        for i in range(0, int(total_intervals), 200):
            intervals.append([i * time_span_seconds + start_date.timestamp(),
                             (i + 200) * time_span_seconds + start_date.timestamp()])
        print(f'intervals: {intervals}')
        return intervals
    else:
        intervals = [[start_date.timestamp(), end_date.timestamp()]]
        print(f'intervals: {intervals}')
        return intervals


def get_dates_and_data_from_latest_file(folder='data'):
    # Create the path to the folder
    folder_path = os.path.join(os.getcwd(), folder)

    # Check if the folder exists
    if not os.path.exists(folder_path):
        # If the folder does not exist, return None
        return None,None,None

    # Get a list of all files in the specified folder
    files = os.listdir(folder_path)

    # Filter the list of files to only include files with the desired format
    files = [file for file in files if file.endswith('.csv') and '_' in file]

    # Sort the list of files by their modification time in descending order
    files.sort(key=lambda file: os.path.getmtime(os.path.join(folder_path, file)), reverse=True)

    # Check if there are any files in the list
    if len(files) > 0:
        # If there are files in the list, get the latest file
        latest_file = files[0]

        # Extract the start and end dates from the file name
        start_date_unix, end_date_unix = latest_file.replace('.csv', '').split('_')

        # Convert the start and end dates from Unix timestamps to datetime objects
        start_date = datetime.datetime.fromtimestamp(int(start_date_unix) / 1000)
        end_date = datetime.datetime.fromtimestamp(int(end_date_unix) / 1000)

        # Create the path to the latest file
        latest_file_path = os.path.join(folder_path, latest_file)

        # Load the data from the latest file into a DataFrame
        df = pd.read_csv(latest_file_path)
        
        # Return the start and end dates and the data as a tuple
        return end_date, datetime.datetime.now(), df
    else:
        # If there are no files in the list, return None
        return None,None,None




def fetch_ohlcv_range( start_date, end_date, symbol, interval, category):
    # Get the intervals for the given date range and interval
    intervals = get_intervals(start_date, end_date, interval)

    # Create an empty list to store the data
    data = []

    # Loop through each interval and make an API call for that interval
    for interval_start, interval_end in intervals:
        # Convert the start and end times to Unix timestamps in milliseconds
        start_ts = int(interval_start * 1000)
        end_ts = int(interval_end * 1000)

        print(
            f"Fetching historical data from {datetime.datetime.fromtimestamp(interval_start)} to {datetime.datetime.fromtimestamp(interval_end)}")

        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': start_ts,
            'endTime': end_ts,
        }

        try:
            response = exchange.fetch_ohlcv(symbol, interval, params=params)
            if response:
                data += response
            else:
                print(f"No data returned for this interval")
        except ccxt.BaseError as e:
            print("An error occurred while making the request:")
            print(e)
            print("Retrying after 1 minute...")
            time.sleep(60)  # Wait for 1 minute before retrying
            # Recursive call to retry fetching data
            return fetch_ohlcv_range(exchange, start_date, end_date, symbol, interval)

    # Convert the data to a DataFrame
    df_new = pd.DataFrame(data, columns=[
                        'Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
    # df_new['Date'] = pd.to_datetime(df_new['Date'])
    # df_new.set_index('Date',inplace = True)

    print("returning new")
    print(df_new)
  
    # Convert the start and end dates to Unix timestamps in seconds
    start_date.timestamp()
    end_date.timestamp()
    df_new = df_new[~df_new.index.duplicated(keep='first')]

    return df_new



# %%
# def get_wallet_balance(test, coin):

    
def get_free_balance(symbol: str) -> float:
        # Set up the exchange with your API key and secret
    

    # Fetch the balance for the specified symbol
    balance = exchange.fetch_balance()
    return balance[symbol]['free']
def get_value(symbol: str) -> float:
    # Fetch the balance for the specified symbol
    balance = exchange.fetch_balance()
    return balance[symbol]['free']


def get_market_bid_price(symbol: str) -> float:
    ticker = exchange.fetch_ticker(symbol)
    return float(ticker['bid'])



def get_market_ask_price(symbol: str) -> float:
    ticker = exchange.fetch_ticker(symbol)
    return float(exchange.price_to_precision(symbol,  ticker['ask']))


# get_market_ask_price(True, "BTCUSDT")
# %%
def place_order_tp_sl(testmode, type, symbol, side, tp, sl, amount):
    logger(symbol, side, tp, sl, amount)
    try:
        # Get the market data

        # Get the market price

        exchange.load_markets()
        amount = exchange.amount_to_precision(symbol,amount)
        market = exchange.market(symbol)
        buyresponse = exchange.create_market_order(
            market['id'],side,
            amount
        )
        print(buyresponse)
      
        logger(buyresponse)
        
    # Save the order details to a CSV file
        with open('orders.csv', mode='a') as file:
            writer = csv.writer(file)

            # Write the header row if the file is empty
            if file.tell() == 0:
                writer.writerow(ORDERCOLUMNS)

            # Write the order details
            writer.writerow([
                buyresponse['clientOrderId'],
                datetime.datetime.now(),
                symbol,
                side,
                buyresponse['amount'],
                buyresponse['price'],
                tp,
                sl,
                "",
                "",
                ""
            ])
            return buyresponse
    except Exception as e:
        logger(f"An error occurred while placing the order: {e}")
        return None


def cancel_order(symbol, id):
    try:
        exchange.cancel_order(id,TRADINGPAIR)

    except Exception as e:
        logger(f"An error occurred: {e}")




async def fetch_spot_balance(exchange):
    balance = await exchange.fetch_balance()
    print("Spot Balance:", balance)

