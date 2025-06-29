import pytest
import pandas as pd
from src.preprocess import load_data, validate_data
import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def validate_data(df):
    assert not df.isnull().values.any(), "Data contains null values"
    assert df.shape[1] == 5, "Unexpected number of columns"
    assert set(df['species'].unique()) == {'setosa', 'versicolor', 'virginica'}    

def test_data_validation():
    df = load_data('data/iris.csv')
    validate_data(df)  # Will raise assertion if data is bad    