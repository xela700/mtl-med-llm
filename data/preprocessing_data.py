"""
Script to preprocess data for model training for both ICD classification
and clinical note summarization.
"""

import re
import logging


logger = logging.getLogger(__name__)

def remove_blank_sections(note: str, sections_headers: list[str] = None) -> str:
    """
    Removed redacted sections from MIMIC data that occur as a result of the
    deindentification process.

    Parameters:
    note (str): clinical note to be stripped of unneeded sections
    sections_headers (list[str]): Sections that will be checked for removal. Base sections from MIMIC will be used if none provided

    Return:
    str: clinical note stripped of blank sections
    """

    if sections_headers is None:
        logger.info("Using default section headers to clean data")
        sections_headers = [
            "Name:", "Unit No:", "Admission Date:", 
            "Discharge Date:", "Date of Birth", "Attending:",
            "Social History:", "Followup Instructions:", "Facility:" 
        ]
    
    lines = note.split("\n")
    cleaned = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line == "":
            i +=1
            continue

        if any(header.lower() in line.lower() for header in sections_headers):
            # Captures and removes sections followed by underscore blanks
            if re.fullmatch(r".*:\s*_+\s*", line):
                i += 1
                continue
            # Captures and removes sections with underscore blanks on the following line
            elif i + 1 < len(lines) and re.fullmatch(r"^_+\s*", lines[i + 1].strip()):
                i += 2
                continue
            # Catch all for a header that should be removed
            else:
                i += 1
                continue
        
        else:
            cleaned.append(lines[i])
            i += 1
    
    return "\n".join(cleaned)


def normalize_deidentified_blanks(text: str) -> str:
    """
    Inserts standard placeholder text inplace of de-identified blanks where applicable

    Parameters:
    text (str): Input text w/ de-identified blanks from MIMIC notes

    Returns:
    str: Text w/ replacement placeholders for training
    """

    replacements = {
        r"\b_{2,}\s*(years?\s*old|y\.?o\.?|y/o|yo)": "<AGE>",
        r"\b_{2,}-years?-old\b": "<AGE>",
        r"\b(mrs?|ms)\.?\s_{2,}": "<PATIENT>",
        r"\b(doctor|dr.?)\s_{2,}": "<DOCTOR>",

    }

    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    text = re.sub(r"\b_{2,}\b", "<REDACTED>", text) # Clean up for remaining unknown de-identified fields
    return text