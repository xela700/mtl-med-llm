"""
Script to preprocess data for model training for both ICD classification
and clinical note summarization.
"""

import re


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

        if any(header.lower() in line.lower() for header in sections_headers):
            # Captures and removes sections followed by underscore blanks
            if re.match(r".*:\s*_+\s*$", line):
                i += 1
                continue
            # Captures and removes sections with underscore blanks on the following line
            elif i + 1 < len(lines) and re.match(r"^_+\s*$", lines[i + 1].strip()):
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