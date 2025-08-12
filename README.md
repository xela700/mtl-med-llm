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

## Setup Instructions

Not yet functional — setup steps will be added as development progresses.

## Citations

Models Used:

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
