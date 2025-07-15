"""
Testing module for both fetching and preprocessing data, to ensure proper outputs are being achieved.
"""

from data import fetch_data
from data.preprocessing_data import ClassificationPreprocessor, SummarizationPreprocessor
from datasets import Dataset
import pandas as pd
import logging
import os

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

    save_path_text = "tests/test_data/classification_raw_data.parquet"

    if not os.path.exists(save_path_text):
        fetch_data.fetch_and_save_query(query=query_text, save_path=save_path_text) # raw data

    raw_data = fetch_data.load_data(load_path=save_path_text)

    raw_data.rename(columns={'discharge_note': 'input', 'icd_codes': 'labels'}, inplace=True)

    print(raw_data.head(1))
    print(raw_data['input'].dtype)

    query_labels = """
    SELECT icd_code, COUNT(*) AS count
    FROM `physionet-data.mimiciv_3_1_hosp.diagnoses_icd`
    WHERE icd_version = 10
    GROUP BY icd_code
    ORDER BY count DESC
    LIMIT 5
    """

    save_path_labels = "tests/test_data/classification_label_data.parquet"

    if not os.path.exists(save_path_labels):
        fetch_data.fetch_and_save_query(query=query_labels, save_path=save_path_labels)  # label data

    label_data = fetch_data.load_data(load_path=save_path_labels)

    label_data.rename(columns={'icd_code': 'labels'}, inplace=True)

    print(label_data.head())

    label_ids = label_data.set_index('labels')['count'].to_dict()
    logger.info("Converting label_data dataset to labels dictionary")

    preprocessor = ClassificationPreprocessor("bert-base-cased", label_ids=label_ids, text_col='input', label_col='labels')
    logger.info("Instantiated classification preprocessor")

    logger.info("Testing Remove Blank Sections")
    raw_data['input'] = raw_data['input'].map(preprocessor.remove_blank_sections)

    logger.info("Testing Normalize Deidentified Sections")
    raw_data['input'] = raw_data['input'].map(preprocessor.normalize_deidentified_blanks)

    logger.info("Testing Classification Section Extraction")
    raw_data['input'] = raw_data['input'].map(preprocessor.classification_extract_sections)

    print(raw_data['input'].head())

if __name__ == "__main__":
    test_classification_input()

