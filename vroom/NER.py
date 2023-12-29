r"""  Package for Named Entity Recognition (NER) tasks.

Authors
--------
 * Nicolas Bataille 2023
 * Adel Moumen 2023
"""

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)
import json
from flair.data import Sentence
from flair.models import SequenceTagger
import os
import re
import nltk


def read_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().rstrip()

    # Remove extra whitespaces using regular expression
    content = re.sub(r"\s+", " ", content)
    return content


def write_json_file(file_path: str, data: list):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def get_entities(
    text: str,
    model: AutoTokenizer,
    tokenizer: AutoModelForTokenClassification,
    device="cpu",
):
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
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
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

def chunk_text_by_sentence(text, batch_size=5):
    # Download the punkt tokenizer if not already present
    nltk.download('punkt')

    # Use nltk to tokenize the text into sentences
    sentences = nltk.sent_tokenize(text)

    # Batch the sentences into 5 sentences per item
    batched_sentences = [' '.join(sentences[i:i + batch_size]) for i in range(0, len(sentences), batch_size)]

    # Return the list of batches of sentences
    return batched_sentences
   
def chunk_text_by_paragraph(text):
    # Use nltk to tokenize the text into paragraphs
    paragraphs = [paragraph.strip() for paragraph in text.split('\n') if paragraph.strip()]

    # Return the list of paragraphs
    return paragraphs

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


def get_entities_from_file(
    input_file_path: str,
    source: str = "Jean-Baptiste/camembert-ner",
    device="cpu",
):
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


def get_positions_of_entities(text):
    """ This function returns the positions of the entities in the text.
    
    Args:
        text (str): The text to tag.

    Returns:
        list: A list of dictionaries representing the entities. Each dictionary contains the keys 'word', 'start', and 'end'.
    """
    words = text.split(" ")
    is_PER_main_context = False
    is_PER_second_context = False
    cpt_second_context = 0
    text = []
    for word in words:

        if word == "<PER>" and is_PER_main_context is False:
            is_PER_main_context = True
            text.append(word)
        elif word == "<PER>" and is_PER_main_context is True:
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

    return " ".join(text)


def merge_special_words(word_list):
    merged_list = []
    current_word = ""

    for word in word_list:
        if word in ["<", "PER", "</", ">", "/"]:
            current_word += word
        else:
            if current_word:
                merged_list.append(current_word)
                current_word = ""
            merged_list.append(word)

    if current_word:
        merged_list.append(current_word)

    return merged_list


def separate_words(text):
    words = re.findall(r"\b\w+\b|[^\w\s]", text)

    return merge_special_words(words)


def get_positions_of_entities(text):
    """ This function returns the positions of the entities in the text.

    Args:
        text (str): The text to tag.

    Returns:
        list: A list of dictionaries representing the entities. Each dictionary contains the keys 'word', 'start', and 'end'.
    """
    words = text.split(" ")

    positions = []
    current_entity = []
    current_entity_start = 0
    current_entity_end = 0
    i = 0
    for word in words:
        if word == "<PER>":
            current_entity = []
            current_entity_start = i
        elif word == "</PER>":
            current_entity_end = i
            word = " ".join(current_entity)

            
            positions.append(
                {
                    "word": word,
                    "start": current_entity_start,
                    "end": current_entity_end,
                }
            )
        else:
            if len(word) > 0:
                if len(word) > 0:
                i += len(word) + 1
            current_entity.append(word)

    
    return positions


def tag_text_with_entities(input_file_path, entity_list):
    """ This function tags the text in the given file with the given entities.
    
    Args:
        input_file_path (str): The path to the input file.
        entity_list (list): A list of entities to tag.

    Returns:
        str: The tagged text.
    """
    input_text = read_file(input_file_path)

    tagged_text = [[]]
    entity_list = sorted(entity_list, key=len, reverse=True)
    
    i = 0
    while i < len(input_text):
        found_entity = None
        
        # Try to match each entity in the list starting from the longest
        for entity in entity_list:
            if input_text[i:i + len(entity)] == entity:
                found_entity = entity
                break

        if found_entity:
            # Start tagging
            tagged_text.append([" <PER> "])
            tagged_text[-1].append(found_entity)
            i += len(found_entity)
            tagged_text[-1].append(" </PER> ")
            tagged_text.append([])
        else:
            tagged_text[-1].append(input_text[i])
            i += 1

    concat = []
    for sublist in tagged_text:
        tmp = "".join(sublist)
        concat.append(tmp)

    return "".join(concat)

def tag_text_with_entities_v2(input_file_path, entity_list):
    input_text = read_file(input_file_path)

    tagged_text = [[]]
    entity_list = sorted(entity_list, key=len, reverse=True)

    i = 0
    while i < len(input_text):
        found_entity = None

        # Try to match each entity in the list starting from the longest
        for entity in entity_list:
            if input_text[i : i + len(entity)] == entity:
                found_entity = entity
                break

        if found_entity:
            # Start tagging
            tagged_text.append([" <PER> "])
            tagged_text[-1].append(found_entity)
            i += len(found_entity)
            tagged_text[-1].append(" </PER> ")
            tagged_text.append([])
        else:
            tagged_text[-1].append(input_text[i])
            i += 1

    concat = []
    for sublist in tagged_text:
        tmp = "".join(sublist)
        concat.append(tmp)

    return "".join(concat)


def set_determinants(name: str, determinant_path: str):
    """
    Takes a name and retunds a list of every possible combinations of determinants with the name.

    Args:
        name (str): The name to use.
        determinant_path (str): The path to the file containing the determinants.

    Returns:
        list: A list of every possible combinations of determinants with the name.
    """
    determinants = []
    with open(determinant_path, "r") as file:
        for line in file:
            line = line.strip()
            if "_" in line:
                line = line.replace("_", " ")
                determinants.append(line + name)
    return determinants


def search_names_with_determinants(
    chunk: str,
    entities: list,
    determinant_path: str = "vroom/utils/determinants.txt",
):
    """
    Searches for names with determinants in the given chunk.

    Args:
        chunk (str): The chunk to search in.
        entities (list): List of entities initially found by NER.
        determinant_path (str): The path to the file containing the determinants.

    Returns:
        list: List of new entities with determinants.
    """
    names_with_determinants = []
    for entity in entities:
        determinants = set_determinants(entity, determinant_path)
        for determinant in determinants:
            if re.search(determinant, chunk, re.IGNORECASE):
                names_with_determinants.append(determinant)
    return names_with_determinants
