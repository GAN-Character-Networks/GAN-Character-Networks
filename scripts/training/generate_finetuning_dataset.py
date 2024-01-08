r""" Generates the processed filed used for fine-tuning the model.

Authors
-------
 * Adel Moumen 2024
"""
import csv
import re

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

from vroom.NER import read_file


def reconstruct_sentence_from_list(tokens):
    """Reconstructs a sentence from a list of tokens.

    Args:
        tokens (list): List of tokens to reconstruct.

    Returns:
        str: Reconstructed sentence.
    """
    reconstructed_sentence = ""
    for token in tokens:
        if token.startswith("▁"):
            token = " " + token[1:]  # Remove the leading ▁ if present
        reconstructed_sentence += token
    return reconstructed_sentence.strip()


def extract_entities_from_sentence(input_sentence):
    """Extracts the entities from a sentence.

    Args:
        sentence (str): The sentence to extract entities from.

    Returns:
        list: List of entities.
    """
    return re.findall(r"@@(.*?)##", input_sentence)


def tokenize(input, tokenizer):
    """Tokenizes a sentence and returns the tokens and labels associated with each token.

    Args:
        input (str): The sentence to tokenize.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.

    Returns:
        str: The input sentence.
        list: The tokens of the sentence.
        list: The labels associated with each token.
    """
    entities = extract_entities_from_sentence(input)
    entity_lists = [tokenizer.tokenize(entity) for entity in entities]
    input_sentence_normalized = input.replace("@@", "").replace("##", "")
    input_tokens = tokenizer.tokenize(input_sentence_normalized)

    labels = [0] * len(input_tokens)
    for entity_list in entity_lists:
        entity_len = len(entity_list)
        for i in range(len(input_tokens) - entity_len + 1):
            if input_tokens[i : i + entity_len] == entity_list:
                labels[i : i + entity_len] = [2] * entity_len

    return input_sentence_normalized, input_tokens, labels


def write_results(output_file, data):
    """Writes the tokenized and labeled data to a CSV file.

    Args:
        output_file (str): Path to the output CSV file.
        data (list): List of tokenized and labeled sentences.
    """
    with open(output_file, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "input", "tokens", "ner_tags"])
        for idx, (sentence, tokens, labels) in enumerate(data):
            writer.writerow([idx, sentence, tokens, labels])


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")

    files = [
        "data/test_set/chapter_1.labeled.@@",
        "data/test_set/chapter_2.labeled.@@",
    ]
    all_data = []
    for input_file in files:
        content = read_file(input_file)
        sentences = sent_tokenize(content)
        data = [tokenize(sentence, tokenizer) for sentence in sentences]
        all_data.extend(data)

    output_file = "data/finetuning_data/train_data.csv"
    write_results(output_file, all_data)
