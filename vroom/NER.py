r"""  Package for Named Entity Recognition (NER) tasks.

Authors
--------
 * Nicolas Bataille 2023
 * Adel Moumen 2023
"""

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    # CamembertModel,
    # CamembertTokenizer,
)
import json
from transformers import pipeline
from flair.data import Sentence
from flair.models import SequenceTagger
import os


def read_file(file_path: str):
    with open(file_path, "r") as f:
        return f.read()


def write_json_file(file_path: str, data: list):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def get_entities(text: str):
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
        entities (list): A list of entities with the corresponding BIO tag for each words.

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


def write_bio_tag_file(input_file_path: str, output_file_path: str):
    """
    Tags the text in the given file with BIO tags and writes the tagged text to the output file.

    Args:
        input_file_path (str): The path to the input file.
        output_file_path (str): The path to the output file.
    """
    tagged_text = tag_file(input_file_path)

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, "w") as f:
        f.write(tagged_text)


def write_pos_tag_file(input_file_path: str, output_file_path: str):
    """
    Tags the text in the given file with pos tags and writes the tagged text to the output file.

    Args:
        input_file_path (str): The path to the input file.
        output_file_path (str): The path to the output file.
    """
    with open(input_file_path, "r") as f:
        text = f.read()

    model = SequenceTagger.load("qanastek/pos-french")
    sentence = Sentence(text)
    model.predict(sentence)

    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path))

    with open(output_file_path, "w") as file:
        tokens = [token.text for token in sentence.tokens]
        pos_tags = [token.labels[0].value for token in sentence.tokens]

        result = " ".join(
            [f"{token} <{pos_tag}>" for token, pos_tag in zip(tokens, pos_tags)]
        )

        file.write(result)


def get_entities_from_file(file_path: str):
    """
    Extracts named entities from the given file.

    Args:
        file_path (str): The path to the input file.

    Returns:
        list: A list of list dictionaries representing the named entities for each text chunk. Each dictionary contains the keys 'entity_group',
              'word', 'start', and 'end'.
        list: A list of chunks of the text.
    """
    with open(file_path, "r") as f:
        text = f.read()

    chunks = chunk_text(text)
    entities = []
    for chunk in chunks:
        entities.append(get_entities(chunk))

    return entities, chunks
