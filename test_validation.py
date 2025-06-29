import pytest
import pandas as pd
import pandas as pd

import unittest
import joblib
import pandas as pd

class TestModel(unittest.TestCase):
    restored_df = None
    
    def setUp(self):
        self.restored_df = pd.read_csv(path)

    def validate_data(df):
        assert not self.restored_df.isnull().values.any(), "Data contains null values"
        assert self.restored_df.shape[1] == 5, "Unexpected number of columns"
        assert set(self.restored_df['species'].unique()) == {'setosa', 'versicolor', 'virginica'}    

    def test_data_validation():
        validate_data(df)  # Will raise assertion if data is bad    

if __name__ == '__main__':
    unittest.main()   