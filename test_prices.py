import yfinance as yf
import unittest

class TestDataRetrieval(unittest.TestCase):
    def test_data_retrieval(self):
        msft = yf.Ticker("MSFT")
        data = msft.history(period="1d")
        self.assertTrue(len(data) > 0, "Data retrieval failed")


class TestStockPricePrediction(unittest.TestCase):
    def test_prediction(self):
        # simulate historical data for the last 10 days
        historical_data = [100, 105, 110, 115, 120, 125, 130, 135, 140, 145]
        predicted_price = predict_stock_price(historical_data)
        self.assertTrue(predicted_price > 0, "Prediction failed")