import datetime

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

import threading
import time

from IBJts.samples.Python.Testbed.ContractSamples import ContractSamples


class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []  # Initialize variable to store candle

    def historicalData(self, reqId, bar):
        print(f'Time: {bar.date} Close: {bar.close}')
        self.data.append([bar.date, bar.close])


def run_loop():
    app.run()


app = IBapi()
app.connect('127.0.0.1', 4002, 123)

# Start the socket in a thread
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

time.sleep(1)  # Sleep interval to allow time for connection to server

# Create contract object
contract = Contract()
contract.symbol = 'SPY'
contract.secType = 'STK'
contract.exchange = 'ARCA'
contract.currency = 'USD'

# Request historical candles
# app.reqHistoricalData(4102, ContractSamples.Index(), '', '2 D', '1 hour', 'BID', 0, 2, False, [])
#queryTime = (datetime.datetime.today() - datetime.timedelta(days=180)).strftime("%Y%m%d %H:%M:%S")
#app.reqHistoricalData(4102, ContractSamples.USStock(), queryTime,
 #                     "1 M", "1 day", "MIDPOINT", 1, 1, False, [])
eurgbp_contract = Contract()
eurgbp_contract.symbol = "VOO"
eurgbp_contract.secType = "STK"
eurgbp_contract.currency = "USD"
eurgbp_contract.exchange = "ARCA"
class App(EWrapper, EClient):

    # Receive historical bar data
    def historicalData(self, reqId, bar):
        print("HistoricalData. ReqId:", reqId, "BarData.", bar)

# Request historical bar data
# app.reqHistoricalData(tickerId, contract, endDateTime, durationString, barSizeSetting, whatToShow, useRTH, formatDate, keepUpToDate)
app.reqHistoricalData(1, eurgbp_contract, '', '2 Y', '1 Hour', 'MIDPOINT', 1, 1, False, None)

time.sleep(5)  # sleep to allow enough time for data to be returned

# Working with Pandas DataFrames
import pandas

df = pandas.DataFrame(app.data, columns=['DateTime', 'Close'])
df['DateTime'] = pandas.to_datetime(df['DateTime'])
df.to_csv('VOO_Hourly.csv')

print(df)

app.disconnect()
