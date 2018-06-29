import unittest

from sklearn.linear_model import LinearRegression

from transformers import LabelsClipper
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TestClipLabels(unittest.TestCase):
    def test_transform(self):
        # generate train data
        x = np.random.rand(1000)
        y = 100 * x - 50
        x_test = np.random.rand(1000, 1)
        x = np.expand_dims(x, 1)
        #  transform
        clipped_model = LabelsClipper(LinearRegression())
        clipped_model.fit(x, y)
        predictions = clipped_model.predict(x_test)

        # check results for integers from 1 to 8
        self.assertTrue(issubclass(predictions.dtype.type, np.integer))
        self.assertTrue(( (predictions <= 8) & (predictions >= 1)).all())


if __name__ == "__main__":
    unittest.main()