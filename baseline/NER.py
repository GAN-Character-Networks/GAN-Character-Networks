from transformers import AutoTokenizer, AutoModelForTokenClassification
import json
from transformers import pipeline


def read_file(file_path: str):
    with open(file_path, "r") as f:
        return f.read()


def get_entities(text: str):
    """
    Extracts named entities from the given text.

    Args:
        text (str): The input text.

    Returns:
        list: A list of dictionaries representing the named entities. Each dictionary contains the keys 'entity_group',
              'word', 'start', and 'end'.
    """
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
    model = AutoModelForTokenClassification.from_pretrained(
        "Jean-Baptiste/camembert-ner"
    )
    nlp = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )

    raw_result = nlp(text)
    raw_result = [
        {
            "entity_group": entity["entity_group"],
            "word": entity["word"],
            "start": entity["start"],
            "end": entity["end"],
        }
        for entity in raw_result
    ]

    filtered_results = [
        entity for entity in raw_result if entity["entity_group"] == "PER"
    ]

    return filtered_results


def write_json_file(file_path: str, data: list):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
