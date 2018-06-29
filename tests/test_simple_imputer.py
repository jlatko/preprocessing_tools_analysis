import unittest
from imputers import FillNaTransformer
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class TestSimpleImputer(unittest.TestCase):
    def test_transform(self):
        before = pd.DataFrame({
            'a': np.random.normal(100, 10, 1000),
            'b': np.random.rand(1000),
            'c': [1]*1000,
        })
        # add nan values
        before['a'][5] = np.NaN
        before['b'][10] = np.NaN
        a_mean = before['a'].mean()
        b_median = before['b'].median()
        nan_c_ids = before['c'].sample(n=100).index
        before['c'][nan_c_ids] = np.NaN

        #  transform
        imputer = FillNaTransformer(mean=['a'], median=['b'], zero=['c'], nan_flag=['a'])
        imputer.fit(before)
        after = imputer.transform(before)

        # test output
        self.assertTrue((after['c'][nan_c_ids] == 0).all())
        self.assertEqual(after.isnull().sum().sum(), 0)
        self.assertEqual(after['a'][5], a_mean)
        self.assertEqual(after['b'][10], b_median)
        self.assertIn('a_nan', after.columns)
        self.assertNotIn('b_nan', after.columns)
        self.assertNotIn('c_nan', after.columns)


if __name__ == "__main__":
    unittest.main()