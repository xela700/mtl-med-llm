# Modular Multi-Model Biomedical LLM

This project aims to construct a multi-model LLM capable of handling distinct biomedical NLP tasks (ICD-10 code classification, clinical note summarization, etc.). 
This project leverages publicly accessible LLMs available on Hugging Face as initial models for fine-tuning and testing.

A major goal is to allow modular development on individual tasks by using targeted downstream LLMs specific to each task. An intent classifier head routes input to these models by labeling the user's intent from their prompt.

## Project Status

This project is currently in development. Data used for training restricted, as it requires credentialed access. Demoing of the inference pipeline can be achieved by downloading the correct model components, as described below.

Current NLP tasks: ICD-10 code classification and clinical note summarization

## Features Planned

- [x] Create demo available to pull from repository
- [x] Creation of an interface for inference
- [ ] Adding new biomedical NLP tasks

## Technologies Used

- Python 3.11.5
- Hugging Face Transformers

## Dataset

This project using tables from the larger MIMIC-IV datasets retrievable via credentialed access on PhysioNet. Datasets are not publicly available. Credentialed access to both MIMIC-IV base and MIMIC-IV-note are required to analyze project components at this time. Exact queries for data usage are located in config/config.yaml.

## Environment Setup Instructions

1. **Clone the repository**

    ```bash
    git clone https://github.com/xela700/mtl-med-llm.git
    cd mtl-med-llm

2. **Create Virtual Environment**

    (Only needed once.)

    python -m venv venv

3. **Activate Virtual Environment**

    Linux/macOS:

    source venv/bin/activate

    Windows:

    source venv/Scripts/activate

4. **Install Dependencies**

    pip install -r requirements.txt

## Documentation

Pre-built HTML documentation is included in this repository.

To view it locaclly, open: doc/_build/html/index.html in any web browser.

## Rebuild Documentation (Optional)

If you modify documentation source files or want to regenerate the docs, you
can rebuild them using Sphinx (included in requirements.txt)

Steps from the project root:

```bash
cd doc
python build_docs.py
```

## Demo Setup

1. Download [zip file](https://drive.google.com/drive/folders/1doxwTRtQGf887UEatIBVnhQ7jCc_1hRP?usp=drive_link) containing model components.

2. Run from project root:

```bash
python scripts/demo_setup.py
```

3. Start the API:

```bash
uvicorn api:app
```

4. Once the application has finished startup, it can be found by copying "http://127.0.0.1:8000/" to any browser.

## Citations

**Models Used:**

DistilBERT
distilbert/distilbert-base-uncased
V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “Distilbert, a distilled version of bert: smaller,
faster, cheaper and lighter,” ArXiv, vol. abs/1910.01108, 2019.

BioBERT
dmis-lab/biobert-large-cased-v1.1
J. Lee, W. Yoon, S. Kim, D. Kim, S. Kim, C. H. So, and J. Kang, “Biobert: a pre-
trained biomedical language representation model for biomedical text mining,” Bioinformat-
ics, vol. 36, pp. 1234–1240, 2020.

BioBART
GanjinZero/biobart-v2-large
H. Yuan, Z. Yuan, R. Gan, J. Zhang, Y. Xie, and S. Yu, “Biobart: Pretraining
and evaluation of a biomedical generative language model,” 2022. [Online]. Available:
https://arxiv.org/abs/2204.03905


**Data**

PhysioNet:
 A. L. Goldberger, L. A. N. Amaral, L. Glass, J. M. Hausdorff, P. C.
Ivanov, R. G. Mark, J. E. Mietus, G. B. Moody, C.-K. Peng, and H. E.
Stanley, “Physiobank, physiotoolkit, and physionet: Components of a
new research resource for complex physiologic signals,” Circulation, vol.
101, no. 23, pp. e215–e220, jun 2000, [Circulation Electronic Pages].
[Online]. 
Available: http://circ.ahajournals.org/content/101/23/e215.ful

MIMIC-IV:
A. E. W. Johnson, L. Bulgarelli, L. Shen, A. Gayles, A. Shammout,
S. Horng, T. J. Pollard, S. Hao, B. Moody, B. Gow, L.-w. H. Lehman,
L. A. Celi, and R. G. Mark, “Mimic-iv, a freely accessible electronic
health record dataset,” Scientific Data, vol. 10, 2023.

MIMIC-IV (3.1):
A. Johnson, L. Bulgarelli, T. Pollard, B. Gow, B. Moody, S. Horng,
L. A. Celi, and R. Mark, “Mimic-iv (version 3.1),” 2024. [Online].
Available: https://doi.org/10.13026/kpb9-mt58

MIMIV-IV-Note:
A. Johnson, T. Pollard, S. Horng, L. A. Celi, and R. Mark,
“Mimic-iv-note: Deidentified free-text clinical notes (version 2.2),”
2023. [Online]. 
Available: https://doi.org/10.13026/1n74-ne17
