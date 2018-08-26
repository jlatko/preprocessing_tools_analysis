import unittest
from transformers import OutliersClipper
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TestOutliersClipper(unittest.TestCase):
    def test_transform(self):
        before = pd.DataFrame({
            'a': np.random.normal(100, 10, 1000),
            'b': np.random.rand(1000),
            'c': [0]*1000,
            'd': np.random.randint(-5,5,1000),
            'e': np.random.rand(1000),
        })

        # manually add outliers
        before['e'][0] = 5
        before['e'][1] = -5
        before['d'][2] = 100
        before['d'][3] = -100
        before['b'][4] = 5
        before['b'][5] = -5
        before['a'][4] = 0
        before['a'][5] = 1000

        # save means and deviations
        means = before.mean()
        stds = before.std()
        mins = means - 3 * stds
        maxes = means + 3 * stds

        # fit and transform
        clipper = OutliersClipper(['a', 'b', 'c', 'd'])
        clipper.fit(before)
        after = clipper.transform(before.copy())

        # test output
        pd.testing.assert_index_equal(before.index, after.index)
        pd.testing.assert_series_equal(before['c'], after['c'])
        pd.testing.assert_series_equal(before['e'], after['e'])
        self.assertEqual(after['d'][2], maxes['d'])
        self.assertEqual(after['d'][3], mins['d'])
        self.assertFalse((after['d'] > maxes['d']).any())
        self.assertFalse((after['d'] < mins['d']).any())
        self.assertFalse((after['a'] > maxes['a']).any())
        self.assertFalse((after['a'] < mins['a']).any())
        self.assertFalse((after['b'] > maxes['b']).any())
        self.assertFalse((after['b'] < mins['b']).any())


if __name__ == "__main__":
    unittest.main()