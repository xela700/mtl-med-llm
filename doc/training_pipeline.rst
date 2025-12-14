Training Pipeline
=================

This document outlines the source code used to train the various LLMs in this project,
including classification, summarization, and intent detection models. It includes details regarding
modifications to parameters, configurations, and adapters used during training.

Also included are the metrics and evaluation functions used to assess model performance during training, 
as well as the custom Callbacks used to improve GPU usage and log metrics.

Source Code
-----------

.. literalinclude:: ../model/train_model.py
    :language: python
    :linenos:

.. literalinclude:: ../model/evaluate_model.py
    :language: python
    :linenos: