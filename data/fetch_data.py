"""
Script to fetch data from BigQuery for use in model fine tuning.

Primary data sourced from MIMIC-IV and is not intended for public access.   
"""

from utils.config_loader import load_config
from google.cloud import bigquery
import os
import pandas as pd
import logging
from pyarrow import ArrowInvalid

logger = logging.getLogger(__name__)


def fetch_and_save_query(query: str, save_path: str = None) -> None:
    """
    Creates a save path if none exists/provided. Uses input query to pull from BigQuery.
    Data saved to parquet file in save path.

    Parameters:
    query (str): a SQL query
    save_path (str): optional path to save return from query to

    Raises:
    FileNotFoundError: if the provided path is not valid
    ValueError: if the provided query cannot be used by BigQuery
    """

    project_id = 'fine-tuned-med-llm' # Google Cloud project connected to LLM construction

    client = bigquery.Client(project=project_id)
    
    if not validate_query(query=query, client=client):
        logger.error("Provided query is invalid.")
        raise ValueError("Provided query is invalid.")

    if save_path is None:
        config = load_config()
        save_path = config["data"]["data_path"]

    elif os.path.exists(save_path):
        logger.info(f"Warning: {save_path} exists and will be overwritten with new query.")

    try:
        query_job = client.query(query=query)
        df = query_job.to_dataframe()
        df.to_parquet(save_path)
        logger.info(f"Saved data to {save_path}")
    except (ArrowInvalid, ValueError, OSError) as err:
        logger.error(f"Data failed to save: {err}")
        raise ValueError(f"Invalid parquet file format at {load_data}") from err



def validate_query(query: str, client: bigquery.Client) -> bool:
    """
    Helper function for fetch_and_save_query(). Checks that the input query follows
    SQL syntax by performing a dry run before executing proper.
    Will return True if the query is valid.

    Parameters:
    query (str): a SQL query
    bigquery.Client: client connection to Google BigQuery

    Returns:
    bool: whether BigQuery recognizes the query as valid
    """
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)

    try:
        client.query(query=query, job_config=job_config)
        return True
    except Exception as e:
        logger.warning(f"Query unrecognized by BigQuery")
        return False


def load_data(load_path: str = None) -> pd.DataFrame:
    """
    Loads data from parquet file. Uses config laid out in YAML if no
    path provided.

    Parameters:
    load_path (str): Option path to parquet file

    Returns:
    pd.DataFrame: dataframe from parquet file

    Raises:
    FileNotFoundError: if file doesn't exist
    ValueError: If file is not a valid parquet file
    """
    
    if load_path is None:
        config = load_config()
        load_path = config["data"]["data_path"]

    try:
        df = pd.read_parquet(load_path)
        logger.info(f"Loaded data from {load_path}")
        return df
    except FileNotFoundError as fnf:
        logger.error(f"Data load failed: {fnf}")
        raise
    except (ValueError, ArrowInvalid) as err:
        logger.error(f"Invalid parquet file format as {load_path}: {err}")
        raise ValueError(f"Invalid parquet file format at {load_data}") from err
        
    
