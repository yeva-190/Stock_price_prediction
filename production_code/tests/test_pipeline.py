import unittest
import numpy as np
import pandas as pd
from pipeline import load_data, preprocess_data, train_linear_regression, train_lstm, train_rnn, train_gradient_boosting


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame(np.random.randn(100, 5), columns=['Open', 'High', 'Low', 'Close', 'Volume'])

    def test_load_data(self):
        data = load_data()
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        train_data, test_data = preprocess_data(self.data)
        self.assertIsInstance(train_data, np.ndarray)
        self.assertIsInstance(test_data, np.ndarray)
        self.assertEqual(train_data.shape[0], int(len(self.data) * 0.8))
        self.assertEqual(test_data.shape[0], len(self.data) - int(len(self.data) * 0.8))

    def test_train_linear_regression(self):
        train_data, _ = preprocess_data(self.data)
        lr_model = train_linear_regression(train_data)
        self.assertIsInstance(lr_model, LinearRegression)

    def test_train_lstm(self):
        train_data, _ = preprocess_data(self.data)
        lstm_model = train_lstm(train_data)
        self.assertIsInstance(lstm_model, Sequential)

    def test_train_rnn(self):
        train_data, _ = preprocess_data(self.data)
        rnn_model = train_rnn(train_data)
        self.assertIsInstance(rnn_model, Sequential)

    def test_train_gradient_boosting(self):
        train_data, _ = preprocess_data(self.data)
        gbr_model = train_gradient_boosting(train_data)
        self.assertIsInstance(gbr_model, GradientBoostingRegressor)


if __name__ == '__main__':
    unittest.main()
