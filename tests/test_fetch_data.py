"""
Module to test functions that fetch and load data from BigQuery
"""

import pytest
import os
import pandas as pd
from google.cloud import bigquery
from data import fetch_data
from pyarrow import ArrowInvalid

project_id = 'fine-tuned-med-llm'

client = bigquery.Client(project=project_id)


def test_validate_query_valid():
    
    query = """
    SELECT *
    FROM `physionet-data.mimiciii_demo.admissions`
    LIMIT 1
    """

    assert fetch_data.validate_query(query=query, client=client)


def test_validate_query_invalid():

    query = "lorem ipsum"

    assert not fetch_data.validate_query(query=query, client=client)


def test_fetch_and_save_query_valid():
    
    query = """
    SELECT *
    FROM `physionet-data.mimiciii_demo.admissions`
    LIMIT 1
    """
    save_path = "tests/data/test_data.parquet"

    fetch_data.fetch_and_save_query(query=query, save_path=save_path)

    assert os.path.exists(save_path)


def test_fetch_and_save_query_bad_query():
    
    query = "lorem ipsum"
    save_path = "tests/data/test_data.parquet"

    with pytest.raises(ValueError):
        fetch_data.fetch_and_save_query(query=query, save_path=save_path)


def test_fetch_and_save_query_bad_path():

    query = """
    SELECT *
    FROM `physionet-data.mimiciii_demo.admissions`
    LIMIT 1
    """
    save_path = "invalid<>file.parquet"

    with pytest.raises((ValueError, ArrowInvalid, OSError)):
        fetch_data.fetch_and_save_query(query=query, save_path=save_path)


def test_load_data_valid():

    load_path = "tests/data/test_data.parquet"

    df = fetch_data.load_data(load_path=load_path)

    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_bad_path():

    load_path = "tests/data/testy_data.parquet"

    with pytest.raises(FileNotFoundError):
        fetch_data.load_data(load_path=load_path)


def test_load_data_wrong_ftype():

    load_path = "tests/data/empty_data.csv"

    with pytest.raises((ArrowInvalid, ValueError)):
        fetch_data.load_data(load_path=load_path)

