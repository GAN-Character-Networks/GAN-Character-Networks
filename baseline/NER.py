from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    # CamembertModel,
    # CamembertTokenizer,
)
import json
from transformers import pipeline

# import torch


def read_file(file_path: str):
    with open(file_path, "r") as f:
        return f.read()


def write_json_file(file_path: str, data: list):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def remove_duplicates_by_word(data):
    """
    Removes duplicate entities based on the 'word' attribute.

    Args:
        data (list): A list of dictionaries representing the entities.

    Returns:
        list: A list of dictionaries with duplicate entities removed.
    """
    unique_dicts = []
    words = set()

    for entity in data:
        word = entity["word"]
        if word not in words:
            unique_dicts.append(entity)
            words.add(word)

    return unique_dicts


def get_entities(text: str, keep_duplicates: bool = False):
    """
    Extracts named entities from the given text.

    Args:
        text (str): The input text.
        keep_duplicates (bool, optional): Whether to keep duplicate entities. Defaults to False.

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

    # Only keep the entities that are persons
    filtered_results = [
        entity for entity in raw_result if entity["entity_group"] == "PER"
    ]

    # Remove duplicates
    if keep_duplicates:
        filtered_results = remove_duplicates_by_word(filtered_results)

    return filtered_results


def add_bio_tags(entities):
    """
    Adds BIO tags to the named entities.

    Args:
        entities (list): A list of dictionaries representing the named entities. Each dictionary contains the keys 'entity_group',
                          'word', 'start', and 'end'.

    Returns:
        list: A list of dictionaries representing the named entities with the added 'bio_tag' key.
    """
    for entity in entities:
        word = entity["word"]
        bio_tag = ""
        for i, token in enumerate(word.split()):
            if i == 0:
                bio_tag += token + "<B-PER>"
            else:
                bio_tag += " " + token + "<I-PER>"
        entity["bio_tag"] = bio_tag
    return entities


def chunk_text(text):
    """
    Chunk the text into a list of subtexts of a size of 500 tokens.

    Args:
        text (str): The input text.

    Returns:
        list: A list of subtexts.
    """
    chunked_text = []
    chunk = []
    words = text.split()
    for word in words:
        if len(" ".join(chunk)) + len(word) > 500:
            chunked_text.append(" ".join(chunk))
            chunk = []
        chunk.append(word)
    if chunk:
        chunked_text.append(" ".join(chunk))
    return chunked_text


def tag_text(text: str, entities: list):
    """
    Tags the text with BIO tags based on the given entities.

    Args:
        text (str): The input text.
        entities (list): A list of dictionaries representing the named entities. Each dictionary contains the keys 'entity_group',
                          'word', 'start', 'end', and 'bio_tag'.

    Returns:
        str: The tagged text.
    """
    tagged_text = ""
    start = 0
    for entity in entities:
        tagged_text += text[start : entity["start"] + 1] + entity["bio_tag"]
        start = entity["end"] + 1
    tagged_text += text[start:]

    return tagged_text


def tag_file(input_file_path: str):
    """
    Tags the text in the given file with BIO tags.

    Args:
        input_file_path (str): The path to the input file.

    Returns:
        str: The tagged text.
    """
    with open(input_file_path, "r") as f:
        text = f.read()

    chunks = chunk_text(text)
    tagged_chunks = []
    for chunk in chunks:
        entities = add_bio_tags(get_entities(chunk))
        tagged_chunks.append(tag_text(chunk, entities))
    return " ".join(tagged_chunks)


def write_tag_file(input_file_path: str, output_file_path: str):
    """
    Tags the text in the given file with BIO tags and writes the tagged text to the output file.

    Args:
        input_file_path (str): The path to the input file.
        output_file_path (str): The path to the output file.
    """
    tagged_text = tag_file(input_file_path)
    with open(output_file_path, "w") as f:
        f.write(tagged_text)


# TODO: method to generate NER list on the whole textfile


# def tokenize(entities: list):
#     model_name = 'camembert-base'
#     tokenizer = CamembertTokenizer.from_pretrained(model_name)
#     model = CamembertModel.from_pretrained(model_name)
