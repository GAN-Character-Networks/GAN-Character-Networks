r"""  Pipeline for cooccurences extraction with baseline methods

Authors
--------
 * Nicolas Bataille 2023
"""
from vroom.NER import get_entities_from_file
from vroom.cooccurences import get_cooccurences


def get_cooccurences_from_text(path: str):
    """
    Removes duplicate entities based on the 'word' attribute.

    Args:
        path (str): The path of the text file.

    Returns:
        list: A list of tuples representing the interactions between entities in the text.
    """

    entities, chunks = get_entities_from_file(path)
    return get_cooccurences(chunks, entities)
