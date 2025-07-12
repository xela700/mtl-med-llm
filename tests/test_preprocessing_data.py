"""
Module to test functions that preprocess data prior to model training
"""

from data.preprocessing_data import ClassificationPreprocessor

def test_remove_blank_sections_default_headers():

    preprocessor = ClassificationPreprocessor("bert-base-cased", {}, "foo", "bar")

    input = "Name: ___ Unit No: ___\nDescription: Lorem ipsum\nSocial History:\n___"

    expected = "Description: Lorem ipsum"

    actual = preprocessor.remove_blank_sections(note=input)

    assert actual == expected

def test_remove_blank_sections_custom_headers():

    custom_headers = ["Hats", "Shirts"]

    preprocessor = ClassificationPreprocessor("bert-base-cased", {}, "foo", "bar", remove_sections=custom_headers)

    input = "Hats: fedora\n\n\nPants: jeans\nShirts:\n___\nShoes:\nsandals"

    expected = "Pants: jeans\nShoes:\nsandals"

    actual = preprocessor.remove_blank_sections(note=input)

    assert actual == expected

def test_normalize_deidentified_blanks():

    preprocessor = ClassificationPreprocessor("bert-base-cased", {}, "foo", "bar")

    input = "Mr. ___, a ___ y/o man was admitted yesterday and was seen by doctor ___."

    expected = "<PATIENT>, a <AGE> man was admitted yesterday and was seen by <DOCTOR>."

    actual = preprocessor.normalize_deidentified_blanks(input)

    assert actual == expected

def test_normalize_deidentified_blanks_unknown_fields():

    preprocessor = ClassificationPreprocessor("bert-base-cased", {}, "foo", "bar")

    input = "___ experienced by Ms. ___ indicates a continuing occurance of ___."

    expected = "<REDACTED> experienced by <PATIENT> indicates a continuing occurance of <REDACTED>."

    actual = preprocessor.normalize_deidentified_blanks(input)

    assert actual == expected

def test_remove_normalize_blanks_process():

    preprocessor = ClassificationPreprocessor("bert-base-cased", {}, "foo", "bar")

    input = "Name: ___ Unit No: ___\nDescription: ___ experienced by Ms. ___ indicates a continuing occurance of ___.\nSocial History:\n___"

    expected = "Description: <REDACTED> experienced by <PATIENT> indicates a continuing occurance of <REDACTED>."

    actual = preprocessor.remove_blank_sections(input)
    actual = preprocessor.normalize_deidentified_blanks(actual)

    assert actual == expected

def test_classification_extraction_default_sections():

    preprocessor = ClassificationPreprocessor("bert-base-cased", {}, "foo", "bar")

    input = "Name: Frank\nDischarge Diagnosis: Headache\nLocation: Canada"

    expected = "Headache"

    actual = preprocessor.classification_extract_sections(text=input)

    assert actual == expected


def test_classification_extraction_custom_sections():

    sections = ["Hats", "Boots"]

    preprocessor = ClassificationPreprocessor("bert-base-cased", {}, "foo", "bar", extract_sections=sections)

    input = "Name: Frank\nDischarge Diagnosis: Headache\nHats: Fedora.\nLocation: Canada\n\nboots: clogs"

    expected = "Fedora. clogs"

    actual = preprocessor.classification_extract_sections(text=input)

    assert actual == expected