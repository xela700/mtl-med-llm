"""
Module to test functions that preprocess data prior to model training
"""

from data import preprocessing_data

def test_remove_blank_sections_default_headers():

    input = "Name: ___ Unit No: ___\nDescription: Lorem ipsum\nSocial History:\n___"

    expected = "Description: Lorem ipsum"

    actual = preprocessing_data.remove_blank_sections(note=input)

    assert actual == expected

def test_remove_blank_sections_custom_headers():

    custom_headers = ["Hats", "Shirts"]

    input = "Hats: fedora\n\n\nPants: jeans\nShirts:\n___\nShoes:\nsandals"

    expected = "Pants: jeans\nShoes:\nsandals"

    actual = preprocessing_data.remove_blank_sections(note=input, sections_headers=custom_headers)

    assert actual == expected

def test_normalize_deidentified_blanks():

    input = "Mr. ___, a ___ y/o man was admitted yesterday and was seen by doctor ___."

    expected = "<PATIENT>, a <AGE> man was admitted yesterday and was seen by <DOCTOR>."

    actual = preprocessing_data.normalize_deidentified_blanks(input)

    assert actual == expected