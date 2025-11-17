"""
Module to test routing pipeline for ensemble model inference
"""

from model.model_inference import intent_prediction, classification_prediction, summarization_prediction, model_routing_pipeline

def test_intent_inference():
    """
    Tests intent inference pipeline to ensure that output will correctly route to a model
    """

    input = "Summarize these notes: The patient complains of chest pain that started earlier today. They rate the pain as a 7/10 and describe it as a sharp, stabbing sensation. They deny any previous history of similar symptoms.\n" \
    "Vital signs are normal (130/80mmHg). A physical exam reveals tenderness upon palpation in the chest area. No other adnormalities noted.\n" \
    "Initial assessment is possible musculoskeletal injury chest pain. Recommend physical rest and aspirin for pain management. Follow up in two weeks if symptoms persist or worsen."

    expected = "summarization"

    actual = intent_prediction(input)

    assert actual == expected

