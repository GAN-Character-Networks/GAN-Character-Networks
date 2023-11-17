r"""  Package for co-occurences tasks.

Authors
--------
 * Nicolas Bataille 2023
"""

import nltk


def set_entities_indexes(text_chunks: list, entities: list):
    """
    Sets the start and end indexes of the entities in the joined text.

    Args:
        text_chunks (list): A list of text chunks.
        entities (list): A list of dictionaries representing the named entities. Each dictionary contains the keys 'entity_group',
                          'word', 'start', and 'end'.

    Returns:
        list: A list of dictionaries representing the named entities. Each dictionary contains the keys 'entity_group',
              'word', 'start' and 'end'.
    """
    start = 0
    for i in range(len(text_chunks)):
        chunk = text_chunks[i]
        for entity in entities[i]:
            entity["start"] += start
            entity["end"] += start
        start += len(chunk) + 1
    return entities


def get_cooccurences(text_chunks: list, entities: list):
    """
    Extracts co-occurences from the given text.

    Args:
        chunks_text(list)): The list of subtexts
        entities (list): The list of list of dictionnaries of entities in their given chunk text.

    Returns:
        list: A list of tuples of entities.
    """
    interactions = []
    text = " ".join(text_chunks)
    # TODO: Manage the aliases here
    entities = set_entities_indexes(text_chunks, entities)
    entities = [entity for sublist in entities for entity in sublist]

    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            if entities[i]["word"] != entities[j]["word"]:
                substring = text[entities[i]["end"] : entities[j]["start"]]
                chunked_substring = nltk.word_tokenize(substring)
                if len(chunked_substring) <= 25:
                    interactions.append(
                        (entities[i]["word"], entities[j]["word"])
                    )

    return interactions
