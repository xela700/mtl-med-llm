from google.cloud import bigquery
import pandas as pd

project_id = 'fine-tuned-med-llm'

client = bigquery.Client(project=project_id)

query = """
SELECT *
FROM `physionet-data.mimiciv_3_1_hosp.admissions`
LIMIT 1
"""

query_job = client.query(query)
df = query_job.to_dataframe()
print(df)

