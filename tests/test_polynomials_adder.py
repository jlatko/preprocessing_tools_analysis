import unittest
from transformers import PolynomialsAdder
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TestPolynomialsAdder(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'a': np.random.normal(100, 10, 1000),
            'b': np.random.rand(1000),
            'c': [2]*1000,
            'd': np.random.randint(-5,5,1000),
            'e': np.random.rand(1000),
        })
        self.columns = self.data.columns
        self.poly = PolynomialsAdder(powers_per_column={'a': [2], 'b': [3], 'c': [2,3]})
        self.new_data = self.poly.transform(self.data.copy())


    def test_new_columns(self):
        new_cols = {'a^2', 'b^3', 'c^2', 'c^3'}
        for col in new_cols:
            self.assertIn(col, self.new_data.columns)
        self.assertEqual(set(list(new_cols) + list(self.columns)), set(self.new_data.columns))

    def test_unchanged_indices(self):
        pd.testing.assert_index_equal(self.data.index, self.new_data.index)

    def test_proper_values(self):
        self.assertTrue(all((self.data['a'] ** 2) == (self.new_data['a^2'])))
        self.assertTrue(all((self.data['b'] ** 3) == (self.new_data['b^3'])))
        self.assertTrue(all((self.data['c'] ** 2) == (self.new_data['c^2'])))
        self.assertTrue(all((self.data['c'] ** 3) == (self.new_data['c^3'])))



if __name__ == "__main__":
    unittest.main()