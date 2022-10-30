from enum import Enum
import time

#'open', 'high', 'low', 'close', 'avg_price', 'ohlc_price', 'oc_diff'
class Candle:
    def __init__(self, date, open, high, low, close, avg_price, ohlc_price, oc_diff):
        self.date = date
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.avg_price = avg_price
        self.ohlc_price = ohlc_price
        self.oc_diff = oc_diff

        def __repr__(self):
            return self.date.strftime() + ": " + str(self.open) + ", " + str(self.high) + ", " + str(
                self.low) + ", " + str(self.close)

        def __str__(self):
            return self.date.strftime() + ": " + str(self.open) + ", " + str(self.high) + ", " + str(
                self.low) + ", " + str(self.close)


class CandleTime(Enum):
    minute = 1
    hour = 2
    day = 3


def arrayToCandle(array):
    print(array)
    return Candle(time.time(), array[0], array[2], array[3], array[4], array[5], array[6], array[7])