r"""  Package for Named Entity Recognition (NER) tasks.

Authors
--------
 * Nicolas Bataille 2023
 * Adel Moumen 2023
"""

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline
)
import json
from flair.data import Sentence
from flair.models import SequenceTagger
import os
import re

def read_file(file_path: str): 
    content = ""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().rstrip()

    # Remove extra whitespaces using regular expression
    content = re.sub(r'\s+', ' ', content)
    return content


def write_json_file(file_path: str, data: list):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

def get_entities(text: str, model: AutoTokenizer, tokenizer: AutoModelForTokenClassification, device = "cpu"):
    """ Extracts named entities from the given text.

    Args:
        text (str): The input text.
        model: The model to use for the NER task.
        tokenizer: The tokenizer to use for the NER task.
        device: The device to use for the NER task.

    Returns:
        list: A list of dictionaries representing the named entities. Each dictionary contains the keys 'entity_group',
              'word', 'start', and 'end'.
    """
    nlp = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device = device
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
        if len(entity["word"]) > 1
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
                bio_tag += "<PER> " + token
            else:
                bio_tag += " " + token
        entity["bio_tag"] = bio_tag + " </PER> "
    return entities


def chunk_text(text, chunk_size=500):
    """
    Chunk the text into a list of subtexts of a size of chunk_size words.

    Args:
        text (str): The input text.
        chunk_size (int): The size of the chunks.

    Returns:
        list: A list of subtexts.
    """
    chunked_text = []
    chunk = []
    words = text.split()
    for word in words:
        if len(" ".join(chunk)) + len(word) > chunk_size:
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
        # warning : having + 1 may cause issues when you have a word that ends with a punctuation
        start = entity["end"]
    tagged_text += text[start:]

    return tagged_text


def tag_file(input_file_path: str, source: str = "Jean-Baptiste/camembert-ner"):
    """
    Tags the text in the given file with BIO tags.

    Args:
        input_file_path (str): The path to the input file.
        source (str): The source of the model to use for the NER task.

    Returns:
        str: The tagged text.
    """
    text = read_file(input_file_path)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForTokenClassification.from_pretrained(source)

    chunks = chunk_text(text)
    tagged_chunks = []
    for chunk in chunks:
        entities = add_bio_tags(get_entities(chunk, model, tokenizer))
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
    text = read_file(input_file_path)

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


def get_entities_from_file(input_file_path: str, source: str = "Jean-Baptiste/camembert-ner", device = "cpu"):
    """
    Extracts named entities from the given file.

    Args:
        input_file_path (str): The path to the input file.
        source (str): The source of the model to use for the NER task.

    Returns:
        list: A list of list dictionaries representing the named entities for each text chunk. Each dictionary contains the keys 'entity_group',
              'word', 'start', and 'end'.
        list: A list of chunks of the text.
    """
    text = read_file(input_file_path)

    tokenizer = AutoTokenizer.from_pretrained(source, device=device)
    model = AutoModelForTokenClassification.from_pretrained(source)

    chunks = chunk_text(text)
    entities = []
    for chunk in chunks:
        entities.append(get_entities(chunk, model, tokenizer))

    return entities, chunks

def tag_named_entities(text, named_entities):
    # Sort named entities by length in descending order
    sorted_entities = sorted(named_entities, key=len, reverse=True)

    tagged_text = text

    for entity in sorted_entities:
        # Handling multi-word entities
        entity_words = entity.split()
        if len(entity_words) > 1:
            entity_pattern = r'\b{}\b'.format(re.escape(' '.join(entity_words)))
        else:
            entity_pattern = r'\b{}\b'.format(re.escape(entity))

        # Tagging the entity in the text
        tagged_text = re.sub(entity_pattern, ' <PER> {} </PER> '.format(entity), tagged_text)

    return tagged_text

def remove_nested_tags(text):

    words = text.split(" ")
    is_PER_main_context = False
    is_PER_second_context = False
    cpt_second_context = 0
    text = []
    for word in words:

      if word == "<PER>" and is_PER_main_context == False:
        is_PER_main_context = True
        text.append(word)
      elif word == "<PER>" and is_PER_main_context == True:
        is_PER_second_context = True
        cpt_second_context += 1
      elif word == "</PER>":

        if is_PER_second_context:
          if cpt_second_context > 0:
            cpt_second_context -= 1
            if cpt_second_context == 0:
              is_PER_second_context = False
        else:
          is_PER_main_context = False
          is_PER_second_context = False
          cpt_second_context = 0
          text.append(word)
      else:
        text.append(word)

    return ' '.join(text)

def tag_text_with_entities(input_file_path, entities):
    """
    Tags the text with the given entities.

    Args:
        text (str): The input text.
        entities (list): A list of entities with the corresponding BIO tag for each words.

    Returns:
        str: The tagged text.
    """
    text = read_file(input_file_path)
    
    output = tag_named_entities(text, entities)
    output = remove_nested_tags(output)

    return output
    
