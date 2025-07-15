"""
Testing module for both fetching and preprocessing data, to ensure proper outputs are being achieved.
"""

import data.fetch_data as fetch
from data.preprocessing_data import ClassificationPreprocessor, SummarizationPreprocessor
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def test_classification_input():

    query_text = """
    SELECT h.subject_id, h.hadm_id, n.text AS discharge_note, ARRAY_AGG(h.icd_code) AS icd_codes
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` AS h
    JOIN `physionet-data.mimiciv_note.discharge` AS n 
        ON (h.subject_id = n.subject_id) AND (h.hadm_id = n.hadm_id)
    WHERE h.icd_version = 10
    GROUP BY h.subject_id, h.hadm_id, n.text
    LIMIT 5
    """

    save_path_text = "tests/data/classification_raw_data.parquet"

    fetch.fetch_and_save_query(query=query_text, save_path=save_path_text) # raw data

    raw_data = fetch.load_data(load_path=save_path_text)

    raw_data.rename(columns={'text': 'input', 'icd_code': 'labels'})

    print(raw_data.head(1))

    query_labels = """
    SELECT icd_code, COUNT(*) AS count
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    WHERE icd_version = 10
    GROUP BY icd_code
    ORDER BY count DESC
    LIMIT 5
    """

    save_path_labels = "tests/data/classification_label_data.parquet"

    fetch.fetch_and_save_query(query=query_labels, save_path=save_path_labels)  # label data

    label_data = fetch.load_data(load_path=save_path_labels)

    label_data.rename(columns={'icd_code': 'labels'})

    print(label_data.head())

    label_ids = label_data.set_index('labels')['count'].to_dict()
    logger.info("Converting label_data dataset to labels dictionary")

    preprocessor = ClassificationPreprocessor("bert-base-cased", label_ids=label_ids, text_col='input', label_col='labels')
    logger.info("Instantiated classification preprocessor")

    