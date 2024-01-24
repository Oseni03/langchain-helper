import pandas as pd
import numpy as np
import pytest
from numpy import nan
from product_pipeline import extract, load


# get data
@pytest.fixture(scope='session', autouse=True)
def df():
    # Will be executed before the first test
    df, tbl = extract()
    yield df
    # Will be executed after the last test
    load(df, tbl)
#

# check if column exists
def test_col_exists(df, column):
    assert column in df.columns

# check for nulls
def test_null_check(df):
    columns = list(df.columns)
    for column in columns:
        assert df[column].notnull().all()

# check values are unique
def test_unique_check(df):
    columns = list(df.columns)
    for column in columns:
        assert pd.Series(df[column]).is_unique

# check data type
def test_productkey_dtype_int(df, column, data_type):
    assert (df[column].dtype == data_type)

# check values in range
def test_range_val(df, column, fro, to):
    assert df[column].between(fro,to).any()

# check values in a list
def test_range_val_str(df):
    assert set(df.Color.unique()) == {'NA', 'Black', 'Silver', 'Red', 'White', 'Blue', 'Multi', 'Yellow','Grey', 'Silver/Black'}

