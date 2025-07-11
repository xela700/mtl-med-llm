"""
Script to preprocess data for model training for both ICD classification
and clinical note summarization.
"""

import re
import logging
from abc import ABC, abstractmethod
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


logger = logging.getLogger(__name__)

class TextPreprocessor(ABC):
    """
    Parent class for preprocessing text imported from dataset.

    Attributes:
    checkpoint (str): The associated HuggingFace model. Must match model used in training.
    text_col (str): Column that contains the text to be cleaned and tokenized
    remove_sections (list[str]): Optional section headers used to remove sections of clinical
    notes. Default of 'None' results in removal of sections that will always be blank due to 
    de-identification
    """

    __slots__ = ("tokenizer", "text_col", "remove_sections")

    def __init__(self, checkpoint: str, text_col: str, remove_sections: list[str] = None):
        self.tokenizer = self._create_tokenizer(checkpoint)
        self.text_col = text_col
        self.remove_sections = remove_sections
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    
    def _create_tokenizer(self, checkpoint: str) -> PreTrainedTokenizerBase:
        """
        Initializes the tokenizer to be used in preprocessing.

        Parameters:
        checkpoint (str): Checkpoint to be used to determine the model. Must be the 
        same as the checkpoint used in training.

        Returns:
        PreTrainedTokenizerBase: tokenizer for the model

        Raises:
        ValueError/OSError if checkpoint provided does not exist or is inaccessible
        """

        try:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        except (ValueError, OSError) as err:
            logger.error(f"Unable to access model using checkpoint {checkpoint}")
            raise ValueError(f"Unable to access model using checkpoint {checkpoint}") from err
        
        return tokenizer

    def remove_blank_sections(self, note: str) -> str:
        """
        Removed redacted sections from MIMIC data that occur as a result of the
        deindentification process.

        Parameters:
        note (str): clinical note to be stripped of unneeded sections

        Return:
        str: clinical note stripped of blank sections
        """

        if self.remove_sections is None:
            logger.info("Using default removable sections to remove always de-identified sections.")
            self.remove_sections = [
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

            if any(header.lower() in line.lower() for header in self.remove_sections):
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
    

    def normalize_deidentified_blanks(self, text: str) -> str:
        """
        Inserts standard placeholder text inplace of de-identified blanks where applicable.

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
    
    
    @abstractmethod
    def preprocess_function(self, batch: dict) -> dict[str, list[int]]:
        """
        Preprocessing method that may vary depending on NLP task. To be determined by child classes.
        Intended for use with dataset.map()

        Parameters:
        batch (dict): batch on which the preprocess function will be applied

        Returns:
        dictionary structure expected by dataset.map()
        """
        pass
    


class ClassificationPreprocessor(TextPreprocessor):
    """
    Preprocessor child class for 
    """

    __slots__ = ("label_ids", "extract_sections", "label_col")

    def __init__(self, checkpoint: str, label_ids: dict[str:int], text_col: str, label_col: str, remove_sections: list[str] = None, extract_sections: list[str] = None):
        super().__init__(checkpoint, text_col, remove_sections)
        self.extract_sections = extract_sections
        self.label_ids = label_ids
        self.label_col = label_col

    def classification_extract_sections(self, text: str) -> str:
        """
        Captures relevant sections for ICD code classification. Used to limit the input size of a given sample while minimizing
        context loss. Only intended for classification task and not summarization.

        Parameters:
        text (str): Input discharge notes with extractable sections
        sections (list[str]): sections to keep from text. If none provided, a default set is used

        Returns:
        str: Extracted sections from text joined together 
        """

        if self.extract_sections is None:
            logger.info("Using default sections to extract data for classification.")
            self.extract_sections = ['discharge diagnosis', 'brief hospital course', 'hospital course',
                        'final diagnosis', 'principal diagnosis', 'history of present illness']
        
        parsed_text = {}

        for section in self.extract_sections:
            # Pattern captures everything after the name of the section up until either the next section title or EOF
            pattern = rf'{section.lower()}[\s:\n](.*?)(\n[A-Z][^:\n]*:|\Z)'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                parsed_text[section] = match.group(1).strip()
        
        return ' '.join(filter(None, parsed_text.values()))
    
    def preprocess_function(self, batch: dict[str, list[str]]) -> dict[str, list[int]]:
        """
        Preprocess function to be mapped using dataset specific to classification task.
        To be used with dataset.map() method.

        Parameters:
        batch (dict[list[str]]): batch from dataset

        Returns:
        Modified data in dataset. Text field tokenized based on tokenizer.
        Multi-hot label vectors created using top k labels.
        """

        tokenized_data = self.tokenizer(batch[self.text_col], truncation=True)

        labels = [0] * len(self.label_ids)
        for id in batch[self.label_col]:
            if id in self.label_ids:
                labels[self.label_ids[id]] = 1
        tokenized_data["labels"] = labels

        return tokenized_data
    

    def filter_text_by_label(self, sample: dict[str, list[str]]) -> bool:
        """
        Filtering method to check for the presence of one (or more) ICD labels that are being used during
        training. To be used with dataset.filter() method

        Parameters:
        sample: dataset sample being provided to check for label presence

        Return:
        bool: true if at least one label is present
        """
        icd_codes = sample.get("labels", []) # gets the labels associated with a clinical note, or an empty list if none present
        return any(code in self.label_ids for code in icd_codes)



    



    
