import os
import tempfile
import unittest
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from validation import main, parse_args


class TestValidation(unittest.TestCase):

    def setUp(self):
        self.model_path = 'test_model.pkl'
        self.data_path = 'test_data.csv'
        self.test_data = pd.DataFrame({
            'feature_1': [1, 2, 3, 4],
            'feature_2': [2, 4, 6, 8],
            'label': [0, 0, 1, 1]
        })
        self.test_model = pickle.dumps({
            'model': 'test_model',
            'params': {'param_1': 0.1, 'param_2': 0.2}
        })
        self.args = {
            'model_path': self.model_path,
            'data_path': self.data_path
        }

    def tearDown(self):
        os.remove(self.model_path)
        os.remove(self.data_path)

    def test_main(self):
        # Save test model and data
        with open(self.model_path, 'wb') as f:
            f.write(self.test_model)
        self.test_data.to_csv(self.data_path, index=False)

        # Run main function
        main(parse_args(args=self.args))

        # Assert outputs
        y_true = self.test_data['label']
        y_pred = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 1.0)
        self.assertTrue(np.array_equal(confusion_matrix(y_true, y_pred), np.array([[2, 0], [0, 2]])))


if __name__ == '__main__':
    unittest.main()
