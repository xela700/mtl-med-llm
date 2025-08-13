# Modular Multi-Model Biomedical LLM

This project aims to construct a multi-model LLM capable of handling distinct biomedical NLP tasks (ICD-10 code classification, clinical note summarization, etc.). 
This project leverages publicly accessible LLMs available on HuggingFace as initial models for fine-tuning and testing.

## Project Status

This project is currently in development. Full-pipeline inference not currently available through github cloning. Data used for training restricted, as it requires credentialed access.

Current NLP tasks: ICD-10 code classification and clinical note summarization

## Features Planned

- [ ] Create demo available to pull from repository
- [ ] Creation of an interface for inference
- [ ] Adding new biomedical NLP tasks

## Technologies Used

- Python 3.11.5
- HuggingFace Transformers

## Dataset

This project using tables from the larger MIMIC-IV datasets retrievable via credentialed access on PhysioNet. Datasets are not publicly available. Credentialed access to both MIMIC-IV base and MIMIC-IV-note are required to analyze project components at this time. Exact queries for data usage located in config/config.yaml.

## Setup Instructions

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

5. **Build Documentation (Optional)**

    sphinx-build -b html doc doc/_build/html

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
