"""
The purpose of this module is to write and check temporary data for classification
downstream task. Larger data size intended to be reused for final training.

**FOR CLARITY: THIS IS FOR THE CLASSIFICATION LABELS**
Preprocessing addresses removing clinical notes w/o applicable labels (ICD codes)

Initial data size: 5,000 ICD-10 codes; ~100,000 clinical notes
Temp data size: 50 ICD-10 codes
"""

import logging
import pandas as pd
from utils.config_loader import load_config
from fetch_data import load_data

logger = logging.getLogger(__name__)

def create_temp_classification_labels(dataframe: pd.DataFrame, k_num_codes: int) -> pd.DataFrame:
    """
    Function takes the larger classification label dataset and reduces it to just the top-k number of ICD-10 codes.

    Args:
        dataframe (pd.DataFrame): dataframe with full classification label data
        k_num_codes (int): number of codes to target
    
    Returns:
        pd.DataFrame: modified dataframe with only the top-k labels for filtering the clinical note dataset
    
    Raises:
        ValueError: if k_num_codes > num of possible codes
    """

    if dataframe.shape[0] < k_num_codes:
        logger.error("k_num_codes cannot be more than available ICD-10 codes.")
        raise ValueError("k_num_codes cannot be more than available ICD-10 codes.")

    dataframe = dataframe.head(k_num_codes)

    return dataframe

def check_num_codes(dataframe: pd.DataFrame) -> int:
    """
    Simple function to check the number of labels.

    Args:
        dataframe (pd.DataFrame): dataframe with classification label data
    
    Returns:
        int: number of unique codes
    """

    return dataframe.nunique(dropna=False).iloc[0]

def main(args: list[str]) -> None:
    """
    Main function for data modification. Loads full classification label data from config file, checks its number of codes,
    modifies the number of codes, checks the number of codes again, and saves the new temp labels for preprocessing use.

    Args:
        args (list[str]): command line arguments
    """

    config = load_config()
    label_data = load_data(config["data"]["task_2"]["data_path"])
    temp_data_path = "data/raw_data/temp_code_freq_data.parquet"

    print(check_num_codes(label_data))

    temp_labels = create_temp_classification_labels(label_data)

    print(check_num_codes(temp_labels))

    temp_labels.to_parquet(temp_data_path)
    logger.info(f"Saved temp labels to {temp_data_path}")


if __name__ == "__main__":
    main()

    


