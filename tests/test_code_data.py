"""
Small test code for working with ICD-10 code description data to get it in proper format for preprocessing.
"""
import pandas as pd

def main():

    base_data = pd.read_csv("data/raw_data/icd10cm_codes_2026.txt", header=None, sep='\t')
    print(base_data.shape)
    base_data.columns = ['text']
    base_data[['labels', 'description']] = base_data['text'].str.split(n=1, expand=True)
    print(base_data.shape)
    base_data['labels'] = base_data['labels'].str.lstrip()
    base_data['description'] = base_data['description'].str.lstrip()

    print(base_data['description'].head(10))

if __name__ == "__main__":
    main()
