import unittest
from transformers import FeatureDropper
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TestFeatureDropper(unittest.TestCase):
    def test_transform(self):
        before = pd.DataFrame({
            'a': np.random.normal(100, 10, 1000),
            'b': np.random.rand(1000),
            'c': [0]*1000,
            'd': np.random.randint(-5,5,1000),
            'e': np.random.rand(1000),
        })

        #  transform
        dropper = FeatureDropper(['a', 'b', 'c'])
        after = dropper.transform(before.copy())

        # test output
        pd.testing.assert_index_equal(before.index, after.index)
        pd.testing.assert_series_equal(before['d'], after['d'])
        pd.testing.assert_series_equal(before['e'], after['e'])
        self.assertIn('d', after.columns)
        self.assertIn('e', after.columns)
        self.assertNotIn('a', after.columns)
        self.assertNotIn('b', after.columns)
        self.assertNotIn('c', after.columns)


if __name__ == "__main__":
    unittest.main()