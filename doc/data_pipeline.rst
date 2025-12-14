Data Preprocessing Pipeline
===========================

This document outlines the data preprocessing steps used to convert raw clinical notes into formats
suitable for training LLMs. It also includes code for creating summary targets for training the
summarization model.

Also included are the prompt creator classes and functions used to construct the intention model's dataset.

Source Code
-----------

.. literalinclude:: ../data/preprocessing_data.py
    :language: python
    :linenos:

.. literalinclude:: ../data/intent_prompt_data.py
    :language: python
    :linenos: